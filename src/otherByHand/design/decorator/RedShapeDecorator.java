package otherByHand.design.decorator;

/**
 * 具体装饰类
 * BufferedInputStream
 */
public class RedShapeDecorator extends ShapeDecorator{
    public RedShapeDecorator(Shape decorateShape) {
        super(decorateShape);
    }
    @Override
    public void draw() {
        setRedBorder(decorateShape);
        decorateShape.draw();
    }
    //增添功能
    private void setRedBorder(Shape decoratedShape){
        System.out.println("Border Color: Red");
    }
}
