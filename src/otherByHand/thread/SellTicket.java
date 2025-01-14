package otherByHand.thread;

/**
 * @author yjy
 * @date 2024/4/2
 * @Description 两个线程交替卖一百张票
 */
public class SellTicket {
    static class Ticket{
        private int count = 0 ;
        private int max_count = 100;
        public void sale(){
            while (true){ // 不是卖一张就完事了
                synchronized (this){ // 多线程共享一个ticket对象，同时操作count数据
                    if(count>=max_count){
                        System.out.println("票已经卖完了");
                        break;
                    }
                    count++;
                    System.out.println(Thread.currentThread()+"正在卖第 "+count+" 张票");
                    try {
                        Thread.sleep(100); //模拟买票耗时，要不一个线程直接全跑完了，控制台看不出差别
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    public static void main(String[] args) {
        Ticket ticket = new Ticket();
        Thread thread1 = new Thread(ticket::sale,"窗口1");
        Thread thread2 = new Thread(ticket::sale);
        thread1.start();
        thread2.start();
    }
}
