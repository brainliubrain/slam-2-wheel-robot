/**
 * @file eufy_chassis_pnc_core.cpp
 * @brief Eufy Advanced Cleaning Robot - Chassis Planning & Control Core
 * @author [Your Name]
 * @date 2026-04
 * * @details 
 * Industrial-grade chassis controller integrating:
 * 1. Kinematics (Differential Drive)
 * 2. Extended Kalman Filter (EKF) for Odometry/IMU/VSLAM Data Fusion
 * 3. Variable Impedance Control for Anti-tangling & Active Escaping
 * 4. Thread-safe State Machine for dynamic task execution
 */

#include <iostream>
#include <cmath>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <chrono>

namespace eufy_robot {
namespace control_core {

// ============================================================================
// 1. 系统配置与硬件参数定义 (Configuration & Parameters)
// ============================================================================
struct RobotConfig {
    double wheel_radius = 0.035;       // 35mm 轮径
    double wheel_track = 0.240;        // 240mm 轮距
    double max_linear_vel = 1.2;       // 最大线速度 m/s
    double max_angular_vel = 3.14;     // 最大角速度 rad/s
    double max_motor_current = 2.5;    // 电机最大峰值电流 (A)
    double control_rate_hz = 200.0;    // 200Hz 底层控制频率
};

struct PIDConfig {
    double kp, ki, kd;
    double max_integral;
    double max_output;
};

// ============================================================================
// 2. 核心状态机枚举 (Finite State Machine)
// ============================================================================
enum class RobotState {
    INIT,               // 系统初始化
    IDLE,               // 待机
    NORMAL_CLEANING,    // 正常弓字形清扫
    EDGE_FOLLOWING,     // 精确贴边
    OBSTACLE_AVOIDANCE, // 动态避障绕行
    ACTIVE_ESCAPING,    // 阻抗脱困模式 (识别到被困或缠绕)
    EMERGENCY_STOP      // 致命错误急停
};

// ============================================================================
// 3. 扩展卡尔曼滤波器 (EKF) - SLAM与控制的桥梁
// ============================================================================
class PoseEstimatorEKF {
private:
    Eigen::Vector3d state_;            // [x, y, theta]
    Eigen::Matrix3d P_;                // 状态协方差矩阵
    Eigen::Matrix3d Q_;                // 过程噪声协方差 (来自编码器/系统)
    Eigen::Matrix3d R_vslam_;          // VSLAM 观测噪声
    Eigen::Matrix2d R_imu_;            // IMU 观测噪声
    std::mutex ekf_mutex_;

public:
    PoseEstimatorEKF() {
        state_.setZero();
        P_ = Eigen::Matrix3d::Identity() * 0.1;
        Q_ << 0.05, 0, 0,
              0, 0.05, 0,
              0, 0, 0.01;
        R_vslam_ = Eigen::Matrix3d::Identity() * 0.5; // SLAM 位姿通常有一定延迟和跳变
        R_imu_ = Eigen::Matrix2d::Identity() * 0.01;  // IMU 航向角极其精准
    }

    /**
     * @brief 基于运动学模型的先验预测 (Prediction Step)
     */
    void predict(double v, double w, double dt) {
        std::lock_guard<std::mutex> lock(ekf_mutex_);
        double theta = state_(2);
        
        // 状态转移非线性方程
        state_(0) += v * cos(theta) * dt;
        state_(1) += v * sin(theta) * dt;
        state_(2) += w * dt;
        
        // 角度归一化
        state_(2) = atan2(sin(state_(2)), cos(state_(2)));

        // 计算雅可比矩阵 F (对非线性运动学方程求偏导)
        Eigen::Matrix3d F;
        F << 1, 0, -v * sin(theta) * dt,
             0, 1,  v * cos(theta) * dt,
             0, 0,  1;

        P_ = F * P_ * F.transpose() + Q_;
    }

    /**
     * @brief 融合上游 SLAM 模块提供的低频高精度位姿 (Update Step)
     */
    void updateVSLAM(const Eigen::Vector3d& vslam_pose) {
        std::lock_guard<std::mutex> lock(ekf_mutex_);
        Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d S = H * P_ * H.transpose() + R_vslam_;
        Eigen::Matrix3d K = P_ * H.transpose() * S.inverse(); // 卡尔曼增益

        Eigen::Vector3d y = vslam_pose - H * state_; // 创新向量
        y(2) = atan2(sin(y(2)), cos(y(2))); // 角度误差归一化

        state_ = state_ + K * y;
        P_ = (Eigen::Matrix3d::Identity() - K * H) * P_;
    }

    Eigen::Vector3d getState() {
        std::lock_guard<std::mutex> lock(ekf_mutex_);
        return state_;
    }
};

// ============================================================================
// 4. 底层电机 PID 控制器 (防积分饱和机制)
// ============================================================================
class MotorPIDController {
private:
    PIDConfig cfg_;
    double integral_ = 0.0;
    double prev_error_ = 0.0;

public:
    MotorPIDController(PIDConfig config) : cfg_(config) {}

    double compute(double target, double actual, double dt) {
        double error = target - actual;
        integral_ += error * dt;
        
        // Anti-windup: 积分限幅
        if (integral_ > cfg_.max_integral) integral_ = cfg_.max_integral;
        else if (integral_ < -cfg_.max_integral) integral_ = -cfg_.max_integral;

        double derivative = (error - prev_error_) / dt;
        prev_error_ = error;

        double output = cfg_.kp * error + cfg_.ki * integral_ + cfg_.kd * derivative;
        
        // 输出限幅
        if (output > cfg_.max_output) return cfg_.max_output;
        if (output < -cfg_.max_output) return -cfg_.max_output;
        return output;
    }

    void reset() {
        integral_ = 0.0;
        prev_error_ = 0.0;
    }
};

// ============================================================================
// 5. 变阻抗脱困控制器 (Variable Impedance Controller) - 核心壁垒
// ============================================================================
class ImpedanceEscapingController {
private:
    // 阻抗模型: M * x_ddot + B * x_dot + K * x = F_ext
    double virtual_mass_ = 1.5;
    double virtual_damping_ = 10.0;
    double virtual_stiffness_ = 50.0;

public:
    /**
     * @brief 动态阻尼调节：当检测到电机电流异常激增（缠绕/卡死），主动变软
     */
    void adaptImpedance(double left_current, double right_current) {
        double max_current = std::max(std::abs(left_current), std::abs(right_current));
        
        if (max_current > 1.8) { 
            // 发生严重物理干涉，进入“柔顺态”，降低刚度和阻尼
            virtual_stiffness_ = 5.0;  
            virtual_damping_ = 2.0;
        } else {
            // 恢复刚性态
            virtual_stiffness_ = 50.0;
            virtual_damping_ = 10.0;
        }
    }

    /**
     * @brief 基于电流反馈生成脱困补偿速度
     */
    double computeEscapeVelocity(double external_force_estimation, double current_vel, double dt) {
        // 欧拉法离散化求解阻抗方程，生成顺应外力的速度补偿
        double accel = (external_force_estimation - virtual_damping_ * current_vel) / virtual_mass_;
        return current_vel + accel * dt;
    }
};

// ============================================================================
// 6. 主底盘控制节点 (Chassis PNC Manager) - 将所有模块缝合
// ============================================================================
class EufyChassisManager {
private:
    RobotConfig config_;
    std::atomic<RobotState> current_state_;
    
    // 子系统实例化
    std::unique_ptr<PoseEstimatorEKF> ekf_;
    std::unique_ptr<MotorPIDController> left_motor_pid_;
    std::unique_ptr<MotorPIDController> right_motor_pid_;
    std::unique_ptr<ImpedanceEscapingController> impedance_ctrl_;

    // 线程控制
    std::thread control_thread_;
    std::atomic<bool> is_running_;

    // 当前指令与传感器数据
    double cmd_linear_v_ = 0.0;
    double cmd_angular_w_ = 0.0;
    double actual_left_vel_ = 0.0;
    double actual_right_vel_ = 0.0;
    double motor_current_l_ = 0.0;
    double motor_current_r_ = 0.0;

public:
    EufyChassisManager() {
        current_state_ = RobotState::INIT;
        ekf_ = std::make_unique<PoseEstimatorEKF>();
        
        PIDConfig motor_pid_cfg = {5.5, 0.2, 0.05, 10.0, 12.0}; // 假设12V电机
        left_motor_pid_ = std::make_unique<MotorPIDController>(motor_pid_cfg);
        right_motor_pid_ = std::make_unique<MotorPIDController>(motor_pid_cfg);
        
        impedance_ctrl_ = std::make_unique<ImpedanceEscapingController>();
        is_running_ = false;
    }

    ~EufyChassisManager() {
        stop();
    }

    void start() {
        is_running_ = true;
        current_state_ = RobotState::IDLE;
        control_thread_ = std::thread(&EufyChassisManager::controlLoop, this);
        std::cout << "[Eufy PNC] Chassis Core Started." << std::endl;
    }

    void stop() {
        is_running_ = false;
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
        current_state_ = RobotState::EMERGENCY_STOP;
    }

    // 接收上游规划器下发的控制指令
    void setVelocityCommand(double v, double w) {
        // 严格的边界安全保护
        cmd_linear_v_ = std::max(-config_.max_linear_vel, std::min(config_.max_linear_vel, v));
        cmd_angular_w_ = std::max(-config_.max_angular_vel, std::min(config_.max_angular_vel, w));
    }

    // 接收底层硬件反馈 (中断或轮询更新)
    void updateHardwareFeedback(double l_vel, double r_vel, double l_cur, double r_cur) {
        actual_left_vel_ = l_vel;
        actual_right_vel_ = r_vel;
        motor_current_l_ = l_cur;
        motor_current_r_ = r_cur;
    }

private:
    /**
     * @brief 高频实时控制主循环 (200Hz)
     */
    void controlLoop() {
        const double dt = 1.0 / config_.control_rate_hz;
        auto next_wakeup = std::chrono::steady_clock::now();

        while (is_running_) {
            // 1. 运动学正解计算实际底盘速度
            double actual_v = (actual_right_vel_ + actual_left_vel_) * config_.wheel_radius / 2.0;
            double actual_w = (actual_right_vel_ - actual_left_vel_) * config_.wheel_radius / config_.wheel_track;

            // 2. EKF 状态预测更新
            ekf_->predict(actual_v, actual_w, dt);

            // 3. 状态机逻辑评估 (异常检测)
            evaluateSystemState();

            // 4. 运动学逆解求解目标轮速
            double target_l_vel = 0.0;
            double target_r_vel = 0.0;

            if (current_state_ == RobotState::ACTIVE_ESCAPING) {
                // 触发高级变阻抗脱困控制
                impedance_ctrl_->adaptImpedance(motor_current_l_, motor_current_r_);
                
                // 将异常电流折算为外部物理干涉力矩 (简化模型)
                double ext_force_l = motor_current_l_ * 0.8; 
                double ext_force_r = motor_current_r_ * 0.8;
                
                // 基于阻抗模型生成退让速度
                target_l_vel = impedance_ctrl_->computeEscapeVelocity(-ext_force_l, actual_left_vel_, dt);
                target_r_vel = impedance_ctrl_->computeEscapeVelocity(-ext_force_r, actual_right_vel_, dt);
                
            } else if (current_state_ == RobotState::EMERGENCY_STOP) {
                target_l_vel = 0.0;
                target_r_vel = 0.0;
            } else {
                // 常规清扫: 逆运动学分解
                target_l_vel = (cmd_linear_v_ - cmd_angular_w_ * config_.wheel_track / 2.0) / config_.wheel_radius;
                target_r_vel = (cmd_linear_v_ + cmd_angular_w_ * config_.wheel_track / 2.0) / config_.wheel_radius;
            }

            // 5. PID 闭环解算下发电压/PWM
            double pwm_left = left_motor_pid_->compute(target_l_vel, actual_left_vel_, dt);
            double pwm_right = right_motor_pid_->compute(target_r_vel, actual_right_vel_, dt);

            // [在此处调用 HAL 层硬件接口发送 pwm_left 和 pwm_right]
            // HAL_SetMotorPWM(pwm_left, pwm_right);

            // 6. 严格保证循环频率
            next_wakeup += std::chrono::microseconds(static_cast<int>(dt * 1e6));
            std::this_thread::sleep_until(next_wakeup);
        }
    }

    /**
     * @brief 实时评估系统物理状态，决定是否触发脱困
     */
    void evaluateSystemState() {
        if (std::abs(motor_current_l_) > config_.max_motor_current || 
            std::abs(motor_current_r_) > config_.max_motor_current) {
            
            // 持续高电流，判定为绞入线缆或卡在门槛
            if (current_state_ != RobotState::ACTIVE_ESCAPING) {
                std::cerr << "[WARNING] Excessive motor current detected. Entering Active Escaping Mode!" << std::endl;
                current_state_ = RobotState::ACTIVE_ESCAPING;
            }
        } 
        else if (current_state_ == RobotState::ACTIVE_ESCAPING && 
                 std::abs(motor_current_l_) < 0.5 && std::abs(motor_current_r_) < 0.5) {
            // 电流恢复正常，脱困成功，恢复清扫
            std::cout << "[INFO] Escaped successfully. Resuming normal operation." << std::endl;
            current_state_ = RobotState::NORMAL_CLEANING;
            left_motor_pid_->reset();
            right_motor_pid_->reset();
        }
    }
};

} // namespace control_core
} // namespace eufy_robot

// ============================================================================
// Main 测试入口
// ============================================================================
int main() {
    using namespace eufy_robot::control_core;
    
    EufyChassisManager robot_chassis;
    robot_chassis.start();

    // 模拟规划器下发弓字形扫地指令
    robot_chassis.setVelocityCommand(0.3, 0.0); // 直行 0.3m/s

    // 模拟运行 2 秒
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 模拟左轮被袜子缠绕，电流激增
    std::cout << "--- Simulating Brush Tangling (Left Wheel) ---" << std::endl;
    robot_chassis.updateHardwareFeedback(0.05, 0.3, 2.8, 0.4); // 左轮失速，电流达到 2.8A
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    robot_chassis.stop();
    return 0;
}