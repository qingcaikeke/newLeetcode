package otherByHand.thread;

import java.util.LinkedList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * @author yjy
 * @date 2024/12/6
 * @Description
 */
// 阻塞队列
public class PducerAndCsumer2 {

    public static void main(String[] args) {
        ShareData data = new ShareData();
        new Thread(new Producer(data)::produce).start();
        new Thread(new Consumer(data)::consume).start();
    }

    static class ShareData {
        // 阻塞队列内部有个reentrantlock
        BlockingQueue<Integer> queue = new ArrayBlockingQueue<>(5);
    }

    static class Producer {
        ShareData data;

        public Producer(ShareData data) {
            this.data = data;
        }

        public void produce() {
            for (int i = 1; i <= 10; i++) {
                try {
                    data.queue.put(i);
                    System.out.println("生产者生产了" + i);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    static class Consumer {
        ShareData data;

        public Consumer(ShareData data) {
            this.data = data;
        }

        public void consume() {
            for (int i = 1; i <= 10; i++) {
                try {
                    Integer take = data.queue.take();
                    System.out.println("消费者消费了" + take);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

}
