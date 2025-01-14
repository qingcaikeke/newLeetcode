package otherByHand.thread;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * @author yjy
 * @date 2024/12/5
 * @Description
 */
public class TakeTurnsPrintABC2 {
    static class PrintABC2 {
        private int flag = 1;
        ReentrantLock lock = new ReentrantLock(); // 先创建锁对象，根据锁对象创建condition对象
        Condition conditionA = lock.newCondition(); //
        Condition conditionB = lock.newCondition();
        Condition conditionC = lock.newCondition();

        public void printA() {
            lock.lock(); //先上锁
            try {
                while (flag != 1) { //用while而非if，确保await后再次判断flag，避免虚假唤醒
                    // await 内部是个死循环while，判断条件到来，跳出循环，抢锁，抢到锁返回
                    conditionA.await(); //等待条件A
                }
                System.out.println("A");
                flag = 2;
                conditionB.signal(); //发出唤醒条件B的信号
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                lock.unlock();
            }
        }
        public void printB() {
            lock.lock(); //先上锁
            try {
                while (flag != 2) {
                    conditionB.await();
                }
                System.out.println("B");
                flag = 3;
                conditionC.signal();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                lock.unlock();
            }
        }
        public void printC() {
            lock.lock(); //先上锁
            try { // 注意try位置
                while (flag != 3) {
                    conditionC.await();
                }
                System.out.println("C");
                flag = 1;
                conditionA.signal();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally { //别忘了finally释放锁
                lock.unlock();
            }
        }
    }


    public static void main(String[] args) {
        PrintABC2 printABC2 = new PrintABC2();
        Thread thread1 = new Thread(() -> {
            for (int i = 0; i < 10; i++) {
                printABC2.printA();
            }
        });
        Thread thread2 = new Thread(() -> {
            for (int i = 0; i < 10; i++) {
                printABC2.printB();
            }
        });
        Thread thread3 = new Thread(() -> {
            for (int i = 0; i < 10; i++) {
                printABC2.printC();
            }
        });
        thread1.start();
        thread2.start();
        thread3.start();
    }

}
