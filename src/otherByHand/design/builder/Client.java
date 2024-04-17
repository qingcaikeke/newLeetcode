package otherByHand.design.builder;

public class Client {
    public static void main(String[] args) {
        //无参构造，对象的表示
        HouseBuilder houseBuilder = new HighBuilding();
        //由director去构造对象
        HouseDirector houseDirector = new HouseDirector(houseBuilder);
        houseDirector.constructHouse();
    }

}
