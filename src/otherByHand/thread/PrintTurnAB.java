package otherByHand.thread;

public class PrintTurnAB {
    public static void main(String[] args) {
        SharedData sharedData = new SharedData();

        Thread threadA = new Thread(new MyRunnable(sharedData, "A"));
        Thread threadB = new Thread(new MyRunnable(sharedData, "B"));

        threadA.start();
        threadB.start();
    }
    public static class SharedData {
        private int turn = 0;
        public synchronized void print(String message, int targetTurn) {
            for (int i = 0; i < 5; i++) {
                while (turn != targetTurn) {
                    try {
                        //当你在类的实例方法中调用另一个实例方法时，你可以省略对象引用,等同于this.wait
                        wait();//wait会释放锁，本质上是调用sharedata对象的方法
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
                System.out.println(message);
                turn = 1 - turn;
                notifyAll();
            }
        }
    }

    static class MyRunnable implements Runnable {
        private SharedData sharedData;
        private String message;
        public MyRunnable(SharedData sharedData, String message) {
            this.sharedData = sharedData;
            this.message = message;
        }
        @Override
        public void run() {
            int targetTurn = message.equals("A") ? 0 : 1;
            sharedData.print(message, targetTurn);
        }
    }

}

