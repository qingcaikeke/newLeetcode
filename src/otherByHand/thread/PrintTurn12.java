package otherByHand.thread;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class PrintTurn12 {
    public static void main(String[] args) {
        Print12 print12 = new Print12();

        Thread threadA = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                print12.print1();
            }
        });
        Thread threadB = new Thread(() -> {
            for (int i = 0; i < 5; i++) {
                print12.print2();
            }
        });

        threadA.start();
        threadB.start();
    }

    public static class Print12 {
        ReentrantLock lock = new ReentrantLock();
        Condition condition1 = lock.newCondition();
        Condition condition2 = lock.newCondition();
        private int flag = 0;

        int count=0;
        int max_count=20;
        public void print1() {
            lock.lock();
            try {
                while (flag%2 != 0) {
                    condition1.await();
                }
                System.out.println(Thread.currentThread() + "1");
                flag++;
                condition2.signal();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }finally {
                lock.unlock();
            }
        }
//        标准synchornize写法，起一个循环限制结束，内部先处理故障情况(该线程不应当前时刻执行)，一个循环套wait，然后正常处理notify

//        public synchronized void print1() {
//            while (count<max_count){
//                while (count%2!=0){
//                    try {
//                        wait();
//                    }catch (InterruptedException e){
//                        e.printStackTrace();
//                    }
//                }
//                System.out.println("1");
//                count++;
//                notifyAll();
//            }
//        }
        public void print2() {
            lock.lock();
            try {
                if (flag%2 != 1) {
                    condition2.await();
                }
                System.out.println(Thread.currentThread() + "2");
                flag++;
                condition1.signal();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }finally {
                lock.unlock();
            }
        }
    }


}

