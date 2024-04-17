package otherByHand.design.builder.simple;


// 客户端代码
public class Client {
    public static void main(String[] args) {
        // 使用建造者模式构建电脑
        Computer computer = new Computer.Builder()
                .setCpu("Intel i7")
                .setMemory("16GB")
                .setHardDisk("1TB SSD")
                .build();

        // 获取最终产品的信息
        System.out.println("CPU: " + computer.getCpu());
        System.out.println("Memory: " + computer.getMemory());
        System.out.println("Hard Disk: " + computer.getHardDisk());
    }
}
