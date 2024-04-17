package otherByHand.thread.cookAndFoddie;

public class Desk {//仓库 存放共享资源
//    为了解耦生产者和消费者的关系，通常会采用共享的数据区域，就像是一个仓库
//    生产者生产数据之后直接放置在共享数据区中，并不需要关心消费者的行为
//    消费者只需要从共享数据区中去获取数据，并不需要关心生产者的行为

    //等待唤醒机制
    //是否有食物，来确定哪个线程执行
    public static int foodFlag = 0;
    //总个数
    public static int count = 0;
    //锁对象
    public static Object lock = new Object();
}
