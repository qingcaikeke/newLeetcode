package otherByHand.design.observe;

// 客户端

/**
 * 定义了一种一对多的依赖关系，让多个观察者对象同时监听并被通知某个主题对象的状态变化。
 * 当主题对象发生变化时，它的所有依赖者（观察者）都会收到通知并自动更新。
 */
public class WeatherStation {
    public static void main(String[] args) {
        WeatherData weatherData = new WeatherData();

        // 创建具体观察者
        CurrentConditionsDisplay currentDisplay = new CurrentConditionsDisplay();

        // 注册观察者到主题
        weatherData.addObserver(currentDisplay);

        // 模拟气象数据变化
        weatherData.setMeasurements(80, 65, 30.4f);
    }
}