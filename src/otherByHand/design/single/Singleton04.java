package otherByHand.design.single;
//双从锁，私有成员，私有构造，公有get，两层if，volatile static 修饰成员
//方法级锁的优化（线程安全的懒加载），会更快一点
//volatile是为了禁止指令重排序（分配空间 初始化 指针指向），防止多线程的情况下有的返回null
public class Singleton04 {
    private volatile static Singleton04 instace;
    private Singleton04(){}
    public static Singleton04 getInstace() {
        if(instace==null){
            synchronized (Singleton04.class){
                if(instace==null){//一定注意有两层if
                    instace = new Singleton04();
                }
            }
        }
        return instace;
    }
}
