package otherByHand.design.builder;

public class HighBuilding extends HouseBuilder{
    @Override
    public void buildBasic() {
        System.out.println("打100米地基");
    }

    @Override
    public void buildWalls() {
        System.out.println("砌墙20米");
    }

    @Override
    public void roofed() {
        System.out.println("高档透明屋顶");
    }
}
