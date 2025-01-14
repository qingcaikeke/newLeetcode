package otherByHand.thread;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author yjy
 * @date 2024/12/29
 * @Description
 */
//run1,run2通过不同的思路,实现了一个线程只打印一个数字
//run1:打印完,释放锁,不抢,等下一个线程打印完才被唤醒,才会继续抢 --> 加锁，打印，唤醒其他，把自己挂起
//run2:打印完,睡觉,其他线程抢锁打印,睡醒了,才参与抢锁

//run3:线程1打印1,11,21,更明确;多个runnable对象创建多线程,公用锁和num,只有抢到锁,验证 num%10==id的时候才打印;所以线程要有id属性,构造的时候传进去
//run4:线程1打印1-10
// 先问清要求，是一个线程只能打一个，然后就切换，还是竞争的，一个线程可以打多个，到时间就释放
// 先释放锁还是先唤醒？一定是先唤醒,最后释放锁
// 记住while在synchornize外面
    // 检验代码正确性,看num=101发生什么,能否正常退出
public class t10Print1to100 {
    public static void main(String[] args) {
        PrintNumber_3 printNumber3 = new PrintNumber_3();
        Runnable runnable = printNumber3::run;
        for (int i = 0; i < 10; i++) {
            new Thread(runnable,"thread:" + i).start();
        }
//        PrintNumber printNumber = new PrintNumber();
//        Runnable runnable = printNumber::run1;
//        for (int i = 0; i < 10; i++) {
//            //用一个printNumber对象,创建十个匿名runnable对象,但是他们仍然公用printNumber对象的属性;也就是说除了耗点空间外没有别的问题
//            Thread thread = new Thread(printNumber::run1, "thread:" + i);
//            //公用一个
//            Thread thread = new Thread(runnable, "thread:" + i);
//            thread.start();
//        }

//        Object lock = new Object();
//
//        for (int i = 0; i < 10; i++) {
//            // 10个线程,每个线程一个runnable对象,公用一把锁,公用static类型的待操作变量
//            Thread thread = new Thread(new PrintNumber_2(lock,i,100),"thread:"+i);
//            thread.start();
//        }


    }
    static class PrintNumber_2 implements Runnable{
        private static int num = 1;
        private final Object lock;
        private final int threadId;
        private final int maxNum;
        public PrintNumber_2(Object lock, int threadId, int maxNum){
            this.lock = lock;
            this.threadId = threadId;
            this.maxNum = maxNum;
        }

        @Override
        public void run() {
            while (true){
                synchronized (lock){
                    if(num>maxNum){
                        break;
                    }else {
                        if(threadId==num%10){
                            System.out.println(Thread.currentThread().getName() + ":" + num);
                            num++;
                            lock.notifyAll();
                        }else{
                            try {
                                lock.wait();
                            } catch (InterruptedException e) {
                                throw new RuntimeException(e);
                            }
                        }
                    }
                }

            }
        }
    }
    static class PrintNumber {
        final Object lock = new Object();
        int num = 1;
//      打印完一个数字就wait(释放锁，让出cpu且不抢占，直到被notify)，保证之后换另一个线程执行
//        注意break处一定要notifyAll，否则无法保证所有线程正常退出
        public void run1() {
            while (true) { //先while再加锁
                synchronized (lock) {
                    if (num > 100) {
                        // 必须有，如果只有break，否则一个线程打完100，num变101，唤醒其他线程，把自己挂起，他自己就永远没办法唤醒，因为其他线程不会执行notify
                        // 本质上就是wait了,但不会被notify
                        lock.notifyAll();
                        break;
                    }
                    System.out.println(Thread.currentThread().getName() + ":" + num);
                    num++;
                    lock.notifyAll();//先notify再把自己挂起，notify不会释放锁，wait才会
                    try {
                        lock.wait();
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                }
            }


        }

        // sleep放同步块里面：只有释放锁的同时 时间片结束，其他线程才有机会获得锁。因为时间片结束本身不会释放锁，其他线程切换过来还是拿不到锁；释放锁后时间片没用完，又会回到while的开始处继续执行
//                    结果：一个线程打印多个数字后才切换
        // sleep放同步块外面：释放锁，当前线程睡眠，cpu时间片结束，线程切换，切换后的线程可以拿到锁，进行打印
//                     处于sleep状态也不会去抢锁，因为不是就绪态，本质还是cpu调度某个线程，然后他去看自己能不能获得锁
//                      发现获得不到锁,不会傻傻等待时间片执行完成,而是立刻把自己挂起,等待被唤醒后才会继续抢cpu,然后判断能不能拿到锁
//                    结果：基本上一个线程打印一个数字，就切换下一个线程打印了
        public void run2() {
            while (true) { //先while再加锁
                synchronized (lock) {
                    if (num > 100) {
                        break;
                    }
                    System.out.println(Thread.currentThread().getName() + ":" + num);
                    num++;
                }
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }

        }
    }
    static class PrintNumber_3{
        static AtomicInteger num = new AtomicInteger(1);
        ReentrantLock lock = new ReentrantLock();
        public void run(){
            while (true){
//                lock.lock();//加入lock还是不对,因为一个线程会一直执行,直到时间片用完,同时这样也没啥意义,直接用int不用atomic也行
                if(num.get()>100){
                    break;
                } else {
                    int i = num.getAndIncrement(); //一整句话都是原子的
                    // 但是得到i和打印i两句话之间不是原子的
                    System.out.println(Thread.currentThread().getName()+":"+i);
                }
//                lock.unlock();放在这无法正常退出粗,枷锁后break没解锁
            }
        }
    }
}
