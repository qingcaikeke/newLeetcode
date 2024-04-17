import java.util.Scanner;

// 注意类名必须为 Main, 不要有任何 package xxx 信息
//next不会吸取前后空格，遇空停止
//nextLine吸取前后空格，回车停止,        String s = in.nextLine(); sout(s.charAt(s.length-1))不是回车而是最后一个字符
public class Main {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        // 注意 hasNext (判断后面是否有非法字符，读到/n返回false)和 hasNextLine（读到/n返回true，冉魏还有个空行） 的区别
//        采用hasNextXxxx() 的话，后面也要用nextXxxx():
//        比如前面用hasNextLine()，那么后面要用 nextLine() 来处理输入;
//        后面用 nextInt() 方法的话,那么前面要使用 hasNext()方法去判断.
        while (in.hasNextInt()) { // 注意 while 处理多个 case
            int n = in.nextInt();
            double[] origin = new double[n];
            double[] discount = new double[n];
            for(int i=0;i<n;i++){
                origin[i] = in.nextDouble();
                discount[i] = in.nextDouble();
                if(origin[i]<=0||discount[i]<=0||origin[i]<discount[i]) {
                    System.out.println("error");
                    return;
                }
            }
            int x = in.nextInt();
            int y = in.nextInt();
            if(x<=0||y<=0||x<y){
                System.out.println("error");
                return;
            }
            double res = count(n, origin, discount, x, y);
            System.out.println(String.format("%.2f",res));
        }
    }
    public static double count(int n,double[] origin,double[] discount,int x,int y){
        double res=Double.MAX_VALUE;
        double originSum = 0;
        for(int i=0;i<n;i++){
            originSum+=origin[i];
        }
        if(originSum>=x) res = originSum-y;
        double afterDiscout = 0;
        for(int i=0;i<n;i++){
            afterDiscout += discount[i];
        }
        return Math.min(afterDiscout, res);
    }
}
