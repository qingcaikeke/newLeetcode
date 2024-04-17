package otherByHand.limiter;

/**
 * @author yjy
 * @date 2024/4/6
 * @Description 令牌桶限流
 */
//RateLimiter.create(1);建一个限流器，参数代表每秒生成的令牌数
//tryAcquire(int permits, long timeout, TimeUnit unit)
    //来设置等待超时时间的方式获取令牌，如果超timeout为0，则代表非阻塞，获取不到立即返回。
public class TokenBucket {
    private long lastTokenTime; // 上一次生成令牌的时间戳
    private long capacity; // 桶的容量
    private long rate; // 令牌生成速率
    private long tokens; // 当前令牌数量

    public TokenBucket(long capacity, long rate) {
        this.capacity = capacity;
        this.rate = rate;
        this.tokens = 0;
        this.lastTokenTime = System.currentTimeMillis();
    }

    // 判断是否允许请求
    public synchronized boolean allowRequest() {
        long currentTime = System.currentTimeMillis();//返回阻塞的时间
        long timeElapsed = currentTime - lastTokenTime;
        lastTokenTime = currentTime;
        // 根据时间间隔计算生成的令牌数量，并限制在桶的容量内
        tokens = Math.min(capacity, tokens + timeElapsed * rate);
        // 判断桶中是否有足够的令牌
        if (tokens >= 1) {
            tokens -= 1;
            return true; // 允许请求
        } else {
            return false; // 拒绝请求
        }
    }
}

