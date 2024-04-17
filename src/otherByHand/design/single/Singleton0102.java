package otherByHand.design.single;
//单例用途：网站计数器。应用程序的日志应用。Web项目中的配置对象的读取。数据库连接池。线程池。
//反射能破坏单例，因为可以调用私有构造方法
//可以在构造方法里加一个检测，检测到有实例直接抛异常
//1.懒汉线程不安全 2.懒汉模式线程安全
public class Singleton0102 {
    private static Singleton0102 instance;
    private Singleton0102(){}
    // 懒汉线程安全价格static就行
    // public synchronized static Singleton01 getInstance(){
    public static Singleton0102 getInstance(){
            if(instance==null){
            instance = new Singleton0102();
        }
        return instance;
    }
}
