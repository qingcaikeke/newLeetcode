package otherByHand.thread;

import java.util.concurrent.*;

public class Main {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        Thread thread01  = new CreateThread01();
        //本质上都是 new Thread().start();
        thread01.start();
        Runnable thread02 = new CreateThread02();
        new Thread(thread02).start();

        Callable<Integer> thread03 = new CreateThread03();
        FutureTask<Integer> task = new FutureTask<>(thread03);//futureTask表示要异步执行的任务，提供对执行结果的访问，callable是要执行的任务
        new Thread(task).start();
        System.out.println("等待完成任务");
        int result = task.get();
        System.out.println("任务结果：" + result);
        //4
        Thread thread04 = new Thread(new Runnable() {
            @Override
            public void run() {
                System.out.println("匿名内部类创建线程");
            }
        });
        //5
        ExecutorService executor = Executors.newFixedThreadPool(3);
        for (int i = 0; i < 5; i++) {//提交五个任务
            //executor.submit(() -> System.out.println("线程池创建线程"));
            executor.submit(new Runnable() {
                @Override
                public void run() {
                    System.out.println("线程池创建线程");
                }
            });
        }
        executor.shutdown();
    }

}
