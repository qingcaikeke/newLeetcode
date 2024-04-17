package otherByHand.limiter;

import java.time.LocalDateTime;

/**
 * @author yjy
 * @date 2024/4/6
 * @Description 漏桶限流 桶就相当于一个队列，储存请求，然后以恒定速率处理请求，超过桶容量则拒绝
 */
public class LeakyBucket {
    private long lastLeakTime; // 上一次漏水的时间戳
    private long capacity; // 桶的容量
    private long rate; // 漏水速率
    private long waterLevel; // 当前水位
    public LeakyBucket(long capacity, long rate) {
        this.capacity = capacity;
        this.rate = rate;
        this.waterLevel = 0;
        this.lastLeakTime = System.currentTimeMillis();
    }
    public synchronized boolean allowRequest(int tokens){
        long currentTime = System.currentTimeMillis();
        long timeElapsed = currentTime-lastLeakTime;
        lastLeakTime = currentTime;
        // 计算漏水后水位
        waterLevel = Math.max(0, waterLevel - timeElapsed * rate);
        // 判断当前水位是否可以容纳请求的令牌数量
        if (waterLevel + tokens <= capacity) {
            waterLevel += tokens;
            return true; // 允许请求
        } else {
            return false; // 拒绝请求
        }
    }

}
