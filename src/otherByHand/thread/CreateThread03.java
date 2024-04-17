package otherByHand.thread;

import java.util.concurrent.Callable;

public class CreateThread03 implements Callable<Integer> {//泛型是返回值类型
    @Override
    public Integer call() throws Exception {//函数式接口，有返回值，可以抛异常
        Thread.sleep(300);
        System.out.println(Thread.currentThread().getName() + "实现callable接口创建futuretask，用task创建线程");
        return Integer.MIN_VALUE;
    }
}
