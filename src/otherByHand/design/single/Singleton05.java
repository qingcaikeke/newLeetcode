package otherByHand.design.single;
//静态内部类实现，既安全又懒加载
//利用了final关键字，变量初始化后就不能指向另一个对象
public class Singleton05 {

    private static class SingletonHolder {
        private static final Singleton05 instace = new Singleton05();
    }
    private Singleton05(){}
    //内部类在第一次被调用的时候完成静态属性的加载
    public static Singleton05 getInstance() {
        return SingletonHolder.instace;
    }

}
