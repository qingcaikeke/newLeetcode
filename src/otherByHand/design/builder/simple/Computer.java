package otherByHand.design.builder.simple;

// 产品类
class Computer {
    private String cpu;
    private String memory;
    private String hardDisk;

    // 构造方法私有，防止直接实例化
    private Computer() {}

    // 公有方法用于获取最终产品
    public String getCpu() {
        return cpu;
    }

    public String getMemory() {
        return memory;
    }

    public String getHardDisk() {
        return hardDisk;
    }

    // 静态内部类作为具体建造者
    static class Builder {
        private Computer computer = new Computer();

        public Builder setCpu(String cpu) {
            computer.cpu = cpu;
            return this;
        }

        public Builder setMemory(String memory) {
            computer.memory = memory;
            return this;
        }

        public Builder setHardDisk(String hardDisk) {
            computer.hardDisk = hardDisk;
            return this;
        }

        // 获取最终产品的方法
        public Computer build() {
            return computer;
        }
    }
}
