package otherByHand.thread;

public class CreateThread01 extends Thread{
    @Override
    public void run() {
        System.out.println(getName()+"继承thread创建线程");//不用Thread.currentThread()，直接使用this就能获得当前线程
    }
}
