package otherByHand.thread;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * @author yjy
 * @date 2024/12/5
 * @Description
 */
public class TakeTurnsPrintAB {

    static class PrintAB {
        private final int max_count = 20;
        private int count = 0;

        public synchronized void printA() {
            while (count < max_count) {
                while(count%2!=0){
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                }
                System.out.println("第count个" + count/2 + "A");
                count++;
                notifyAll();
            }

        }

        public synchronized void printB() {
            while (count < max_count) {
                while (count%2!=1){
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                System.out.println("第count个" + count/2 + "B");
                count++;
                notifyAll();
            }
        }
    }


    public static void main(String[] args) {
        // Runnable 接口是个函数式接口，下面创建线程的方式等于new一个runnable匿名对象作为创建thread的参数
        PrintAB printAB = new PrintAB();
        Thread thead1 = new Thread(printAB::printA);
        Thread thead2 = new Thread(printAB::printB);
        thead1.start();//别忘
        thead2.start();
    }


}
