package otherByHand.design.decorator;

/**
 * 抽象装饰类,为什么是抽象类，因为可以有多种包装方式
 * FilterInputStream,构造方法传入inputStream
 */
public abstract class ShapeDecorator implements Shape{//注意也要Implements
    protected Shape decorateShape;
    public ShapeDecorator(Shape decorateShape){
        this.decorateShape = decorateShape;
    }
    public void draw(){
        decorateShape.draw();
    }

}
