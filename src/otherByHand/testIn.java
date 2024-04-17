package otherByHand;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class testIn {
    /**
     * next（）一定要读取到有效字符后才可以结束输入，对输入有效字符之前遇到的空格键、Tab键或Enter键等结束符，next（）方法会自动将其去掉，
     * 只有在输入有效字符之后，next（）方法才将其后输入的空格键、Tab键或Enter键等视为分隔符或结束符。
     * @param args
     */
    /**
     * split n个空格会n-1个,可以用s.split("\\s+");
     * print 和 println：print()输出完毕后不换行，而println()输出完毕后会换行
     * 控制输出的小数点位数 System.out.printf("%.2f",res);  注意是printf不是println
     * @param args
     */
    public static void main(String[] args) {
        System.out.println(Character.toLowerCase(' '));
        in1(args);
        Scanner in = new Scanner(System.in);
        while (in.hasNext()){
            int n  = in.nextInt();
            //nextline从当前位置开始，回车结束
            in.nextLine();//注意吸收回车，因为nextInt只吸收了数字，没吸收回车
            for (int i = 0; i < n; i++) {
                String line = in.nextLine();
                String[] str = line.split("\\s+"); // \s:一个或多个连续的空白字符
                int len = line.length();
                System.out.println(line.charAt(0)+" "+line.charAt(len-1)+len);
            }
        }
    }
    static class ListNode{
        int val;
        ListNode next;
        ListNode(){}
        ListNode(int val){
            this.val = val;
        }
        public ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }
    static void in1(String[] args){
        Scanner in = new Scanner(System.in);
        List<Integer> list  =new ArrayList<>();
        while (in.hasNext()){
            int n = in.nextInt();
            for(int i=0;i<n;i++){
                list.add(in.nextInt());
            }
            ListNode dummy = new ListNode(-1);
            ListNode cur = dummy;
            for(int i=0;i<list.size();i++){
                ListNode temp = new ListNode(list.get(i));
                cur.next = temp;
                cur = temp;
            }
            ListNode head = dummy.next;
            while (head!=null){
                System.out.print(head.val+" ");
                head = head.next;
            }
        }
    }
}
