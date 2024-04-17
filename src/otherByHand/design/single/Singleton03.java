package otherByHand.design.single;
//饿汉，程序启动直接加载
//类的初始化是由ClassLoader负责加锁保证的，并且在类初始化阶段只会被执行一次，因此不会存在多个线程同时创建实例的情况。

public class Singleton03 {
    private static Singleton03 uniqueInstance = new Singleton03();//private
    private Singleton03(){//private
    }
    public static Singleton03 getUniqueInstance() {//注意static public
        return uniqueInstance;
    }
}
