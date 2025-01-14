package otherByHand.thread;

import java.util.concurrent.CountDownLatch;

/**
 * @author yjy
 * @date 2024/12/6
 * @Description
 */
public class Add10Thread {
    public static class AddTask{
        private int tens;
        private int result = 0;
//        public AddTask(int tens,CountDownLatch latch){
        public AddTask(int tens){
            this.tens = tens;
        }

        public int getResult() {
            return result;
        }

        public int add(){
            for (int i = 0; i < 10; i++) {
                System.out.println(Thread.currentThread() + " " + tens + " " + i);
//                System.out.printf("%s %d %d%n", Thread.currentThread(), tens, i);
                result += tens*10+i;
            }
//            latch.countDown();
            return result;
        }
    }

//    public static void main(String[] args) throws InterruptedException {
//        AddTask[] addTasks = new AddTask[10];
//        Thread[] threads = new Thread[10];
//        int sum=0;
//        for (int i = 0; i < 10; i++) {
//            AddTask addTask = new AddTask(i);
//            Thread thread = new Thread(addTask::add);
//            thread.start();
//            threads[i] = thread;
//            addTasks[i] = addTask;
//        }
//        for (Thread thread : threads) {
//            thread.join();
//        }
//        for (AddTask task : addTasks) {
//            sum += task.getResult();
//        }
//        System.out.println(sum);
//    }
    public static void main(String[] args) throws InterruptedException {
        AddTask[] addTasks = new AddTask[10];
        Thread[] threads = new Thread[10];
        int sum=0;
        CountDownLatch latch = new CountDownLatch(10);
        for (int i = 0; i < 10; i++) {
            AddTask addTask = new AddTask(i);
            addTasks[i] = addTask;
            Thread thread = new Thread(() ->{
                addTask.add();      //可以把latch传到addTask的构造方法里，然后在add最后countDown；多个task公用一个latch对象
                latch.countDown(); //每个线程执行完都调用一次countDown
            });
            thread.start();
        }
        latch.await();// 主线程等待所有countDown执行完成
        for (AddTask task : addTasks) {
            sum += task.getResult();
        }
        System.out.println(sum);
    }
}
