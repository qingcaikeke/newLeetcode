package otherByHand.thread;

/**
 * @author yjy
 * @date 2024/4/2
 * @Description 两个线程交替卖一百张票
 */
public class SellTicket extends Thread {
    public static int ticket = 0;
    @Override
    public void run() {
        while (true){
            synchronized (SellTicket.class){
                if(ticket==100) break;
                else {
                    try {
                        Thread.sleep(30);//当前线程睡，而非主线程睡
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    ticket++;
                    System.out.println(getName()+"正在卖"+ticket);
                }
            }
        }
    }

    public static void main(String[] args) {
        SellTicket t1 = new SellTicket();
        SellTicket t2 = new SellTicket();
        t1.setName("窗口1");
        t2.setName("窗口2");
        t1.start();
        t2.start();
    }
}
