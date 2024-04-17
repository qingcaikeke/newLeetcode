package otherByHand.thread.cookAndFoddie;

public class ThreadDemo {
    public static void main(String[] args) {
        Cook c = new Cook();
        foodie f = new foodie();
        c.start();
        f.start();
    }


}
