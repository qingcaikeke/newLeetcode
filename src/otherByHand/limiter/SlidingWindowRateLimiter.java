package otherByHand.limiter;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * @author yjy
 * @date 2024/4/6
 * @Description
 */
public class SlidingWindowRateLimiter {
    private int windowSize; // 窗口大小，单位为秒
    private int maxRequests; // 窗口内允许的最大请求数
    private Deque<Long> requestTimes; // 存储请求的时间戳

    public SlidingWindowRateLimiter(int windowSize, int maxRequests) {
        this.windowSize = windowSize;
        this.maxRequests = maxRequests;
        this.requestTimes = new ArrayDeque<>();
    }

    // 判断是否允许新的请求
    public synchronized boolean allowRequest() {
        long currentTime = System.currentTimeMillis() / 1000; // 转换为秒
        // 移除窗口外的请求时间戳
        while (!requestTimes.isEmpty() && requestTimes.peek() <= currentTime - windowSize) {
            requestTimes.poll();
        }
        // 如果窗口内的请求数超过最大限制，则拒绝请求
        if (requestTimes.size() >= maxRequests) {
            return false;
        }
        // 否则允许请求，并记录当前请求的时间戳
        requestTimes.offer(currentTime);
        return true;
    }
}

