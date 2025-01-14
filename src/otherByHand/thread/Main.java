package otherByHand.thread;

import java.util.Arrays;
import java.util.concurrent.*;

public class Main {
    static class CreateThread03 implements Callable<Integer> {//泛型是返回值类型
        @Override
        public Integer call() throws Exception {//函数式接口，有返回值，可以抛异常
            Thread.sleep(300);
            System.out.println(Thread.currentThread().getName() + "实现callable接口创建futuretask，用task创建线程");
            return Integer.MIN_VALUE;
        }
    }
    public static void main(String[] args) throws ExecutionException, InterruptedException {

        Callable<Integer> thread03 = new CreateThread03();
        FutureTask<Integer> task = new FutureTask<>(thread03);//futureTask表示要异步执行的任务，提供对执行结果的访问，callable是要执行的任务
        FutureTask<Integer> task1 = new FutureTask<>(() -> {
            Thread.sleep(300);
            System.out.println("Ff");
            return Integer.MIN_VALUE;
        });

        new Thread(task).start();
        System.out.println("等待完成任务");
        int result = task.get();
        System.out.println("任务结果：" + result);

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
