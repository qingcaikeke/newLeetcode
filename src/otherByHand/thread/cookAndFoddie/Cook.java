package otherByHand.thread.cookAndFoddie;

public class Cook extends Thread{
    @Override
    public void run() {
        while (true){
            synchronized (Desk.lock){
                if(Desk.count==10) break;
                else {
                    if(Desk.foodFlag==1){
                        try {
                            Desk.lock.wait();
                        } catch (InterruptedException e) {
                            throw new RuntimeException(e);
                        }
                    }
                    else {
                        System.out.println("厨师做了一碗面");
                        Desk.foodFlag = 1;
                        Desk.lock.notifyAll();//唤醒正在等待该锁的线程
                    }
                }
            }
        }
    }
}
