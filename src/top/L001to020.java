package top;

import java.util.*;
//2月1到2月3
public class L001to020 {
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {
        }
        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    public class ListNode {
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
    /**
     * 1.两数之和：一次遍历减少时间（不用一次存map，一次遍历）
     * 2.两数之和：本身逆序，个位是对齐的,所以直接加就行，重点在于别忘了全加完可能还有进位
     * 3.无重复字符的最长字串,类似于自定义滑动窗口，双指针加set实现，优化时间：map
     * 4.寻找两个正序数组的中位数
     * 5.最长回文字串，只想到暴力，没想到动态规划或怎么规划，优化空间复杂度：中心扩散法
     * 6.z字形变换：模拟,两种方式，一种用flag，一种用周期
     * 7.整数翻转：注意爆int
     * 8.字符串转换成整数：模拟
     * 9.回文数：转成string或反转一半数字
     * 10.正则表达式匹配：动态规划，一点不会，子串匹配的题目，不由自主的先想DP
     * 11.盛水最多的容器：双指针
     * 12.整数转罗马数字：模拟，把400啥的也用字母表示出来，没啥价值
     * 13.罗马数字转整数：模拟，当前字符比下一个字符代表的值小就减，否则就加
     * 14.最长公共前缀：二维数组横向扫描 2.写一个公共前缀函数，12比，然后得到的前缀和和3比（两两找公共前缀） 3.在2的基础上使用分治，算左半前缀和再算右半 4.将字符串数组按字典序排序，然后比第一个和最后一个就行
     * 15.三数之和：穷举 -> 排序加双指针，去重是重点，排序保证下一重循环的元素壁上一重大，加了个顺寻，避免重复，双指针将将n^2优化为n，类似11盛水最多容器如果像两数之和。为什么不用map或set？最后去重不方便
     * 16.最接近的三数之和：上边问题的简化，因为可以不去重
     * 17.电话号码的数字组合：回溯搜索
     * 18.四数之和，和三数一样，注意可能爆int，
     * 19.删除链表倒数第n个节点：初始dummy防null；同一起点，快先走n，快到null，慢到倒数第n
     * 20.有效的括号：栈，栈后入先出的特点与括号排序特点一致，后遇到的左括号要先闭合。不能起数组单纯计算数量，因为(}是无效的括号
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode pre = dummy.next;
        int carry = 0;
        while (l1!=null||l2!=null){
            int l1val = l1==null? 0 : l1.val;
            int l2val = l2==null? 0 : l2.val;
            int val = l1val+l2val+carry;
            carry = val/10;
            pre.next = new ListNode(val%10);
            pre = pre.next;
            if(l1!=null) l1 = l1 .next;
            if(l2!=null) l2 = l2.next;
        }
        if(carry!=0) pre.next = new ListNode(carry);
        return dummy.next;
    }
    public int lengthOfLongestSubstring(String s) {
        if(s==null||s.length()==0){
            return 0;
        }
        if(s.length()==1) return 1;
        int left=0,right = 0;
        Set<Character> set = new HashSet<>();
        int maxLen = 0;
        while (right<s.length()){
            //可以用map存下标，这样就能直接定位left，不需要循环，也不需要remove，会更新char储存的下标
            while (set.contains(s.charAt(right))){
                set.remove(s.charAt(left));
                left++;
            }
            set.add(s.charAt(right));
            right++;
            int len = right-left;
            maxLen = Math.max(len,maxLen);
        }
        return maxLen;
    }
    //使用map的
    public int lengthOfLongestSubstring1(String s){
        if (s.length()==0) return 0;
        Map<Character,Integer> map = new HashMap<>();
        int left=0,right=0;
        int maxLen=0;
        while (right<s.length()){
            if(map.containsKey(s.charAt(right))){
                left = Math.max(left,map.get(s.charAt(right))+1);
            }
            map.put(s.charAt(right),right);
            right++;
            int len = right-left;
            maxLen = Math.max(len,maxLen);
        }
        return maxLen;
    }
    public static double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length,n = nums2.length;
        int left = (m+n+1)/2;
        int right = (m+n+2)/2;
        return (getKth(nums1,0,nums2,0,left)+getKth(nums1,0,nums2,0,right))*0.5;
    }
    //两个有序数组找第k个数
    private static int getKth(int[] nums1,int i1,int[] nums2,int i2,int k){
        if(i1==nums1.length){
            return nums2[i2+k-1];
        }
        if(i2==nums2.length){
            return nums1[i1+k-1];
        }
        if(k==1){
            return Math.min(nums1[i1],nums2[i2]);
        }
        int p1 = Math.min(nums1.length,i1+(k/2)) -1;//因为是下标，所以减1
        int p2 = Math.min(nums2.length,i2+(k/2)) -1;
        if(nums1[p1]<nums2[p2]){
            k = k-(p1-i1+1);//本次移除了多少元素
            return getKth(nums1,p1+1,nums2,i2,k);
        }else {
            k = k-(p2-i2+1);
            return getKth(nums1,i1,nums2,p2+1,k);
        }
    }

    public static void main(String[] args) {
        int[] nums1 = new int[]{1,2};
        int[] nums2 = new int[]{3,4};
        findMedianSortedArrays(nums1,nums2);
    }
    public String longestPalindrome(String s) {
        int n = s.length();
        int left = 0,maxLen=0;
        boolean[][]dp = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = true;
        }
        //i是左边界，j是右边界，所以i小于等于j，i+1，j-1在左下
        for(int j=1;j<n;j++){
            for(int i=0;i<j;i++){
                //如果abbc，ij分别等于1，2，i+1就会大于j-1，即超出上半区，要么把下半区也初始化了，要么加一个判断
                dp[i][j] =(i+1>j-1 || dp[i+1][j-1]) && s.charAt(i)==s.charAt(j);
                if(dp[i][j]){
                    int len = j-i+1;
                    if(len>maxLen){
                        maxLen = len;
                        left = i;
                    }
                }
            }
        }
        return s.substring(left,left+maxLen);
    }
    public String convert(String s, int numRows) {
        if(numRows==1) return s;//取余不能为0
        List<StringBuffer> list = new ArrayList<>();
        for(int i=0;i<numRows;i++){
            list.add(new StringBuffer());//注意初始化是add
        }
        int flag = -1,row=0;
        for(char c:s.toCharArray()){
            if(row==0||row==numRows-1) flag=-flag;
            list.get(row).append(c);
            row = row+flag;
        }
        StringBuffer sb = new StringBuffer();
        for(int i=0;i<numRows;i++){
            sb.append(list.get(i));
        }
        return sb.toString();
    }
    public static int reverse(int x) {
        int num = 0;
        //int flag = 1;取余是带正负的，不需要这个判断
        while (x!=0){
            int cur = x%10;
            int lastNum = num;
            num = num*10+cur;
            //判断是否爆int，否则需要用if判断当前是否214748364并且下一位大于7
            if(num/10!=lastNum) return 0;
            x = x/10;
        }
        return num;
    }
    public int myAtoi(String s) {
        //可以s.trim().toCharArray，不过会耗费on空间，因为去掉空格相当于建立了一个新字符串
        int i=0,num=0;
        int flag=1;
        while (s.charAt(i)==' ') i++;
        int start = i;
        for(;i<s.length();i++){
            char c = s.charAt(i);
            //去除空格后的第一位只可能是正负或者数字
            if(c=='+'&&i==start) flag = 1;//防止42+2
            else if(c=='-'&&i==start) flag = -1;
            else if(Character.isDigit(c)){//可以 >='0'&&<='9'
                int cur = c-'0';
                int preNum = num;
                num = num*10+cur;
                if(num/10!=preNum){
                    if(flag==-1) return Integer.MIN_VALUE;
                    else return Integer.MAX_VALUE;
                }
            }
            else break;
        }
        return num*flag;
    }
    public boolean isPalindrome(int x) {
        if(x<0||x%10==0&&x!=0) return false;
        int num=0;
        while (x>num){
            int cur = x%10;
            num = num*10+cur;
            x = x/10;
        }
        return x==num||x==num/10;
    }
    public boolean isMatch(String s, String p) {
        //1.为什么要动态规划：p 中字符 + 星号的组合而言，它可以在 sss 中匹配任意自然数个字符，并不具有唯一性。
        // 因此我们可以考虑使用动态规划，对匹配的方案进行枚举。
        //即a*可以代表一个或多个a，所以会产生与p的字串的匹配问题，即重叠子问题
        //2.注意题意的意思是a*需要看成一个整体，可以当成若干个a，可以当成空
        int m = s.length(),n = p.length();
        char[] ss = s.toCharArray();
        char[] pp = p.toCharArray();
        //因为规划过程中涉及到i-1，j-1，0，m的话会出界
        boolean[][] dp = new boolean[m+1][n+1];
        //初始化较复杂，要注意，已经保证了不会出界，只需把该为true的地方初始化为true
        dp[0][0] = true;//aaa与"ab*a*c*a"
        dp[0][1] = false;
        for(int j=2;j<=n;j++){//题意a*能匹配空，a*b不行
            if(pp[j-1]=='*') dp[0][j] = dp[0][j-2];
        }
        //dp[i][j]：p的[0,j-1]能不能构成s的[0,i-1]个
        for(int i=1;i<=m;i++){
            for (int j=1;j<=n;j++){
                if(pp[j-1]==ss[i-1] || pp[j-1]=='.') dp[i][j] = dp[i-1][j-1];
                else if(pp[j-1]=='*'){
                    //不需要枚举这个组合到底匹配了 sss 中的几个字符
                    //没代表任何元素||匹配s中的一个字符，并把该字符扔掉，p中字符加*的组合可以继续使用
                    if(pp[j-2]=='.'||pp[j-2]==ss[i-1]) dp[i][j] = dp[i][j-2]||dp[i-1][j];
                    else dp[i][j] = dp[i][j-2];//没代表任何字符
                }else dp[i][j] = false;
            }
        }
        return dp[m][n];
    }
    public int maxArea(int[] height) {
        //面积等于min(左右)乘间距，间距每次变小，若想让面积变大，唯一的方法是将较小的变向里挪，即使长边向里变长也不可能比原来面积大
        int left = 0 ,right = height.length-1;
        int area;
        int res = 0;
        while (left<right){
            if(height[left]<height[right]){
                area = height[left]*(right-left);
                left++;
            }else {
                area = height[right]*(right-left);
                right--;
            }
            res = Math.max(res,area);
        }
        return res;
    }
    public int romanToInt(String s) {
        Map<Character, Integer> map = new HashMap<Character, Integer>() {{
            put('I', 1);
            put('V', 5);
            put('X', 10);
            put('L', 50);
            put('C', 100);
            put('D', 500);
            put('M', 1000);
        }};
        int res = 0;
        for(int i=0;i<s.length();i++){
            int val = map.get(s.charAt(i));
            if(i<s.length()-1 && val<map.get(s.charAt(i+1))){
                res = res - val;
            }else {
                res = res+val;
            }
        }
        return res;
    }
    public String longestCommonPrefix(String[] strs) {
        int len0 = strs[0].length();
        int count = strs.length;
        String res = "";
        int i;
        for(i=0;i<len0;i++){
            char c = strs[0].charAt(i);
            for(int j=1;j<count;j++){
                if(i==strs[j].length()||strs[j].charAt(i)!=c){//len0不一定是最短那个，可能会越界
                    res = strs[0].substring(0,i);
                    return res;
                }
            }
        }
        if(i==len0) res = strs[0];
        return res;//可以直接改为return strs[0],因为循环完成没有返回说明，str[0]全读完了
    }
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);int n = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        for(int i=0;i<nums.length-2;i++){
            if(i!=0 && nums[i]==nums[i-1]) continue;
            int target = -nums[i];
            //双指针的作用在于把穷举所有两个元素的组合优化为两个指针的移动，从n^2变为n,因为只有沿着优化方向才能得到好的结果
            int left =i+1,right = n-1;
            while (left<right){
                if(nums[left]+nums[right]==target){
                    res.add(Arrays.asList(nums[i],nums[left],nums[right]));
                    left++;right--;
                    while (left<right && nums[left]==nums[left-1]) left++;
                    while (left<right && nums[right]==nums[right+1]) right--;
                }
                else if(nums[left]+nums[right]<target) left++;
                else right--;
            }
        }
        return res;
    }
    public int threeSumClosest(int[] nums, int target) {
        int res = Integer.MAX_VALUE;
        Arrays.sort(nums);int n = nums.length;
        for(int i=0;i<nums.length-2;i++){
            if(i>0 && nums[i]==nums[i-1]) continue;
            int left =i+1,right = n-1;
            while (left<right){
                int sum = nums[i]+nums[left]+nums[right];
                if(Math.abs(res-target)>Math.abs(sum-target)){
                    res = sum;
                }
                if(sum==target) return res;
                else if(sum<target){//可以加去重
                    left++;
                }else {
                    right--;//可以加去重
                }
            }
        }
        return res;
    }
    public List<String> letterCombinations(String digits) {
        Map<Character, String> map = new HashMap<Character, String>() {{
            put('2', "abc");
            put('3', "def");
            put('4', "ghi");
            put('5', "jkl");
            put('6', "mno");
            put('7', "pqrs");
            put('8', "tuv");
            put('9', "wxyz");
        }};
        List<String> res = new ArrayList<>();
        dfsLetter(res,new StringBuffer(),0,digits,map);
        return res;
    }
    public void dfsLetter(List<String> res,StringBuffer sb,int index,String digits,Map<Character, String> map){
        //出口
        if(index==digits.length()){
            res.add(sb.toString());
            return;
        }
        //1.得到当前的数字
        char c = digits.charAt(index);
        String letters = map.get(c);
        //2.遍历数字对应的字母
        for(int i=0;i<letters.length();i++){
            //3.加到path
            sb.append(letters.charAt(i));
            //4.继续向下
            dfsLetter(res,sb,index+1,digits,map);
            //5.回溯撤销
            sb.deleteCharAt(index);
        }
    }
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        Arrays.sort(nums);
        for(int i=0;i<n-3;i++){
            if(i>0 && nums[i]==nums[i-1]) continue;
            for(int j=i+1;j<n-2;j++){
                if(j!=i+1 && nums[j]==nums[j-1]) continue;
                int left = j+1,right = n-1;
                while (left<right){
                    long sum = (long)nums[i]+nums[j]+nums[left]+nums[right];
                    if(sum==target){
                        res.add(Arrays.asList(nums[i],nums[j],nums[left],nums[right]));
                        left++;right--;
                        while (left<right && nums[left]==nums[left-1]) left++;
                        while (left<right && nums[right]==nums[right+1]) right--;
                    }else if(sum<target){
                        left++;
                    }else {
                        right--;
                    }
                }
            }
        }
        return res;
    }
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0,head);
        //同一起点，快先走k，到null正好慢倒数第k
        ListNode slow = dummy,fast = dummy;
        for(int i=0;i<n;i++){
            fast = fast.next;
        }
        while (fast.next!=null){
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }
    public boolean isValid(String s) {
        Map<Character,Character> map = new HashMap<>(){{
            put('}','{');
            put(')','(');
            put(']','[');
        }};
        Stack<Character> stack = new Stack<>();
        for(char c: s.toCharArray()){
            //如果是右括号，需要和栈顶元素比较
            if(map.containsKey(c)){
                if(stack.isEmpty() || stack.peek()!=map.get(c)) return false;
                stack.pop();
            }else {
                //左括号直接放入
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }

}
