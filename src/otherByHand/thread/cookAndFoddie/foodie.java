package otherByHand.thread.cookAndFoddie;

public class foodie extends Thread{
    @Override
    public void run() {
        /**
         * 1.循环
         * 2.同步代码块
         * 3.判断 ：到了末尾怎么样
         * 4.没到末尾怎么样
         */
        while (true){
            synchronized (Desk.lock){
                if(Desk.count==10) break;
                else {
                    if(Desk.foodFlag==0){//如果没有就等待
                        try {
                            Desk.lock.wait();//让当前线程和锁进行绑定,锁对象有wait和notify方法
                        } catch (InterruptedException e) {
                            throw new RuntimeException(e);
                        }
                    }
                    else {
                        Desk.count++;
                        System.out.println("正在吃第"+Desk.count);
                        Desk.foodFlag = 0;
                        Desk.lock.notify();//唤醒厨师
                    }

                }
            }
        }
    }
}
