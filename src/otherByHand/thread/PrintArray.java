package otherByHand.thread;

/**
 * @author yjy
 * @date 2024/12/6
 * @Description
 */
// 轮流打，所以需要一个flag确保是当前轮次
public class PrintArray {
    public static void main(String[] args) {
        Task task = new Task();
        new Thread(task::print1,"打印数组1").start();
        new Thread(task::print2,"打印数组2").start();
    }
    static class Task{
        int[] arr1 = new int[]{1, 3, 5, 7, 9};
        int[] arr2 = new int[]{2, 4, 6, 8, 10};
        int flag=0;
        public synchronized void print1(){
            for (int i : arr1) {
                while (flag!=0){
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                System.out.println(Thread.currentThread() + Integer.toString(i));
                flag = 1;
                notify();
            }
        }
        public synchronized void print2(){
            for (int i : arr2) {
                while (flag!=1){
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                System.out.println(Thread.currentThread() + Integer.toString(i));
                flag = 0;
                notifyAll();
            }
        }
    }


}
