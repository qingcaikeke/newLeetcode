package otherByHand.design.builder;


import otherByHand.design.entity.House;

//抽象建造者
public abstract class HouseBuilder {
    protected House house = new House();

    public abstract void buildBasic();
    public abstract void buildWalls();
    public abstract void roofed();

    public House buildHouse(){
        return house;
    }
}
