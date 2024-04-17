package otherByHand.limiter;

/**
 * @author yjy
 * @date 2024/4/6
 * @Description
 */
public class FixedWindowRateLimiter {
    private final int windowSize; // 窗口大小，单位为秒
    private final int maxRequests; // 窗口内允许的最大请求数
    private int requestCount; // 窗口内已经发出的请求数
    private long windowStart; // 窗口开始时间

    public FixedWindowRateLimiter(int windowSize, int maxRequests) {
        this.windowSize = windowSize;
        this.maxRequests = maxRequests;
        this.requestCount = 0;
        this.windowStart = System.currentTimeMillis() / 1000; // 转换为秒
    }

    // 判断是否允许新的请求
    public synchronized boolean allowRequest() {
        long currentTime = System.currentTimeMillis() / 1000; // 转换为秒
        // 如果当前时间已经超过了窗口的结束时间，重置窗口开始时间和请求数
        if (currentTime >= windowStart + windowSize) {
            windowStart = currentTime;
            requestCount = 0;
        }
        // 如果窗口内的请求数超过最大限制，则拒绝请求
        if (requestCount >= maxRequests) {
            return false;
        }
        // 否则允许请求，并增加请求数
        requestCount++;
        return true;
    }
}
