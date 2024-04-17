package hot;

import java.util.Scanner;

/**
 * @author yjy
 * @date 2024/4/15
 * @Description
 */
public class RealQuestion {
    public static class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }
    //1. 虾皮旋转链表，力扣原题，求全长取余，找到对应位置断链，再拼接
    //2.小写字母翻转，保持单词相对位置不变，其余非小写字母位置不变，如“Shoppee 1a3c abc”，应翻转为"Seepph 1c3a cba"
    //3.英雄联盟买英雄，背包问题，但要输出买了什么东西，过了dp不到为什么只过了93
    //4.最大数，力扣179原题
    public static String reverses(String str) {
        String[] strings = str.split(" ");
        StringBuffer sb = new StringBuffer();
        for (String string : strings) {
            sb.append(getReverseLowStr(string));
            sb.append(" ");
        }
        sb.deleteCharAt(sb.length()-1);
        return sb.toString();
    }
    private static String getReverseLowStr(String str) {
        //把所有小写字符拿出来，然后翻转，然后再拼接
        StringBuffer tempSb = new StringBuffer();
        for(char c:str.toCharArray()){
            if(c>='a' && c<='z'){
                tempSb.append(c);
            }
        }
        tempSb.reverse();
        int i =0;
        StringBuffer resSb = new StringBuffer();
        for(char c: str.toCharArray()){
            if(c>='a' && c<='z'){
                resSb.append(tempSb.charAt(i));
                i++;
            }else{
                resSb.append(c);
            }
        }
        return resSb.toString();
    }
    //todo leet416
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        String s = in.nextLine();
        System.out.println(reverses(s));
    }

}
