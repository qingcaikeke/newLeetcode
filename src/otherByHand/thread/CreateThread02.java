package otherByHand.thread;

public class CreateThread02 implements Runnable{
    @Override
    public void run() {//无返回值
        System.out.println(Thread.currentThread().getName() + "实现runnable接口创建线程");
    }
}
