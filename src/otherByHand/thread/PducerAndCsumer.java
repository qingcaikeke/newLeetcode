package otherByHand.thread;

import java.util.concurrent.CountDownLatch;

/**
 * @author yjy
 * @date 2024/12/6
 * @Description
 */
// 法2：可用reentrantLock代替
public class PducerAndCsumer {
    public static void main(String[] args) {
        ShareData data = new ShareData();
        CountDownLatch producerLatch = new CountDownLatch(3);
        for (int i = 0; i < 3; i++) {
            new Thread(new Producer(data,producerLatch)::produce, "生产者-" + i).start();
            new Thread(new Consumer(data)::consume, "消费者-" + i).start();
        }
        //生产消费5s后修改data状态，停止生产消费
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        data.stopPduce = true;
        //等待所有生产停止
        try {
            producerLatch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("停止生产");
        data.stopCsume = true;
    }

    static class Producer {
        ShareData data;
        CountDownLatch producerLatch;
        public Producer(ShareData data, CountDownLatch producerLatch) {
            this.data = data;
            this.producerLatch = producerLatch;
        }

        public void produce() {
//            一共要生产10个，桌子上最多有五个
//            for(int i=0;i<10;i++){
//                synchronized (data.lock){
//                    while (data.count==5){
//                        data.lock.wait();
//                    }
//                    System.out.println("生产了一个");
//                    notifyAll();
//                }
//            }
            while (!data.stopPduce) {
                synchronized (data.lock) {
                    while (data.count >= data.MAX_COUNT) {
                        System.out.println(Thread.currentThread() + "发现缓冲区满，等待");
                        try {
                            data.lock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    if (!data.stopPduce) {
                        data.count++;
                        System.out.println(Thread.currentThread() + "生产了一个，当前count " + data.count);
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    data.lock.notifyAll();
                }
            }
            producerLatch.countDown();
        }
    }

    static class Consumer {
        ShareData data;

        public Consumer(ShareData data) {
            this.data = data;
        }

        public void consume() {
            while (true) {
                synchronized (data.lock) {
                    while (data.count == 0 && !data.stopCsume) {
                        System.out.println(Thread.currentThread() + "发现资源为0，等待");
                        try {
                            data.lock.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    if (data.count > 0) {
                        data.count--;
                        System.out.println(Thread.currentThread() + "消费了一个产品，当前数量 " + data.count);
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    data.lock.notify();
                    if (data.count == 0 && data.stopCsume) {
                        break;
                    }
                }
            }
        }
    }

    static class ShareData {
        //如果不想用static，就要main中创建对象，然后生产者和消费者构造函数接收该对象，保证公用
        private int count = 0;
        private final int MAX_COUNT;
        private final Object lock = new Object();
        private boolean stopPduce = false;
        private boolean stopCsume = false;

        public ShareData() {
            MAX_COUNT = 10;
        }

        public void stopProduce() {
            stopPduce = true;
        }

        public void stopConsume() {
            stopCsume = true;
        }

    }

}
