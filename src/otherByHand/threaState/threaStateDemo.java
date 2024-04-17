package otherByHand.threaState;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class threaStateDemo {
    public static void main(String[] args) throws InterruptedException {
        Object object = new Object();
        Thread t1 = new Thread(()->{
            synchronized (object){
                System.out.println("1");
                try {
                    object.wait(1000);
                    System.out.println("2");
                    object.wait();
                    System.out.println("3");
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        });
        t1.start();
        Thread.sleep(1000);//表示主线程
        synchronized (object){
            System.out.println("4");
            object.notify();
            Thread.sleep(1000);
            System.out.println("5");
        }
        Thread.sleep(3000);
        System.out.println("6");
        Thread.sleep(1000);
        System.out.println("7");
        synchronized (object){
            object.notify();
        }
        System.out.println("8");
        Thread.sleep(1000);
        System.out.println("9");
    }
}
