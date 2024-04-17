package top;

import java.util.*;
//2月7到2月13
public class L040toL060 {
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
    static void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    /**
     * 41.
     * 42.接雨水:竖着算，两个数组储存左边最大和右边最大，取二者较小减去当前高，计算一列雨水。优化为双指针没写好，扩展单调栈方法没写好
     * 43.字符串相乘：大数相乘，没想出来 1.按竖式，做加法2.开数组，做乘法，数组右面有几个个数说明他是第几位
     * 44.
     * 45.跳跃游戏2:应该为贪心，想的动态规划，但是复杂度太差了，可能因为子问题重合度不大？
     * 46.全排列：区分组合总和，核心在于used数组
     * 47.全排列二：多了一个横向去重
     * 48.旋转图像：模拟，没模明白
     * 49.字符串异位分词：模拟，把什么存入map？1.字母加个数 2.升序后的结果
     * 50.x的n次方:忘了负数转正数，count一直翻倍，导致栈溢出，两种方式，一种好理解，每次减去2的n次方，但涉及到重复计算
     * 51.
     * 52.
     * 53.最大子数组和：动态规划 或分治（分治不好想）（感觉有点sb）
     * 54.螺旋矩阵:模拟，注意边界条件
     * 55.跳跃游戏：贪心，区分跳跃2，想的复杂了，别搞错if的位置
     * 56.合并区间：lamad表达式，注意边界条件
     * 57.插入区间
     * 58.最后一个单词的长度：处理字符串
     * 59.螺旋矩阵2：模拟，和1差不多
     * 60.
     */
    public int trap(int[] height) {
        int n = height.length;
        int res = 0;
        int left = 1,right = n-2;
        int leftMax = height[0],rightMax = height[n-1];
        while (left<=right){//要有等于，因为如果n=3
            if(leftMax<rightMax){
                if(leftMax>height[left]){
                    res += leftMax - height[left];//找到短板，增加水量
                }
                leftMax = Math.max(leftMax,height[left]);//更新左边最大
                left++;
            }else {
                if(rightMax>height[right]){
                    res+=rightMax-height[right];
                }
                rightMax = Math.max(rightMax,height[right]);
                right--;
            }
        }
        return res;
    }
    public int trap1(int[] height){
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        int res = 0;
        for(int i=1;i<height.length;i++){
            while (height[i]>height[stack.peek()]){//等于的话也入栈，严格大于才出栈
                int bottom = stack.pop();
                if(stack.isEmpty()){//别忘，去取出一个，栈里必须至少还有一个才能接雨水
                    break;
                }
                int left = stack.peek();
                    res += (Math.min(height[left],height[i]) - height[bottom]) * (i-left-1);//不能是i-bottom
            }
            stack.push(i);
        }
        return res;
    }
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int m = num1.length(),n = num2.length();
        int[] arr = new int[m+n];//m位乘n位结果为m+n-1位或m+n位
        for(int i=m-1;i>=0;i--){//核心在于第i位与第j位相乘写在arr[i+j+1](因为数组开大了一位)
            int x = num1.charAt(i) - '0';
            for(int j=n-1;j>=0;j--){
                int y = num2.charAt(j) - '0';
                arr[i+j+1] += x*y;//别忘+=因为可能是x的第2位乘y的第3位，也可能是x3乘y2
            }
        }
        //统一处理进位
        for (int i=m+n-1;i>0;i--){//注意是>0第一位只可能来自进位
            arr[i-1] += arr[i]/10;
            arr[i] = arr[i]%10;
        }
        int begin = arr[0]==0? 1:0;//看最高位有没有数，没有从第一位开始
        StringBuffer sb = new StringBuffer();
        for(int i =begin;i<m+n;i++){
            sb.append(arr[i]);
        }
        return sb.toString();
    }
    public static int jump(int[] nums) {
        //每次在上次能跳到的范围（end）内选择一个能跳的最远的位置
        int maxPosition=0;
        int step=0;
        int end=0;//如果更新end，step就要+1
        for(int i=0;i<nums.length-1;i++){//为什么是len-1？如果最后一个位置恰好是end会增加一次跳跃次数
            int position = i+nums[i];
            maxPosition = Math.max(position,maxPosition);
            if(i==end){//上一次能跳到的最远的位置是end
                //step次一定可以跳到maxPosition
                end = maxPosition;
                step++;
            }
        }
        return step;
    }
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        dfsPer(res,new ArrayList<>(),nums,new boolean[nums.length]);
        return res;
    }
    public void dfsPer(List<List<Integer>> res,List<Integer> path,int[] nums,boolean[] used){
        if(path.size()==nums.length){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=0;i<nums.length;i++){
            if(used[i]) continue;
            used[i] =  true;
            path.add(nums[i]);
            dfsPer(res,path,nums,used);
            used[i] = false;
            path.remove(path.size()-1);
        }
    }
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        dfsPerUni(res,new ArrayList<>(),nums,new boolean[nums.length]);
        return res;
    }
    public void dfsPerUni(List<List<Integer>> res,List<Integer> path,int[] nums,boolean[] used){
        if(path.size()==nums.length){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=0;i<nums.length;i++){
            if(i!=0 && !used[i-1] && nums[i]==nums[i-1]){
                continue;//判断当前元素与上一个元素相等，并且上一个元素没有被使用过，用于横向去重
            }
            if(used[i]) continue;//判断第i个元素有没有被用过，防止元素的重复使用（纵向）
            used[i] =  true;
            path.add(nums[i]);
            dfsPer(res,path,nums,used);
            used[i] = false;
            path.remove(path.size()-1);
        }
    }
    public void rotate(int[][] matrix) {
        //一次循环可以旋转四个元素
        //不是遍历所有的元素，遍历所有会被覆盖，只需要转1/4即可，需要考虑矩阵行数的奇偶
        int n = matrix.length;
        //双重循环遍历1/4个元素
        for(int i=0;i<n/2;i++){
            for(int j=0;j<(n+1)/2;j++){
                //完成四个元素的旋转,记录左上，按左上，左下，右下，右上的顺序填
                int temp = matrix[i][j];
                //因为i,j->j,n-1-i;所以 n-1-j,i-> i,j(换元)
                matrix[i][j] = matrix[n-1-j][i];
                matrix[n-1-j][i] = matrix[n-1-i][n-1-j];
                matrix[n-1-i][n-1-j] = matrix[j][n-1-i];
                matrix[j][n-1-i] = temp;
            }
        }
        return;
    }
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String,List<String>> map = new HashMap<>();
        for(String str:strs){
            char[] s = str.toCharArray();
            //打一个频次表作为key可能更快一点，一个是k，一个是klogk
            Arrays.sort(s);
            String key = Arrays.toString(s);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(str);
            map.put(key,list);//别忘了新list需要put
        }
        return new ArrayList<>(map.values());
    }
    public double myPow(double x, int N) {
        long n = N;
        if(n==0) return 1;
        // 负数情况,需要注意-MIN转正数的溢出(-MIN取反后结果不变)
        if (n < 0) {
            x = 1 / x;
            n = -n;
        }
        double res = 1.0 ;
        //x的n次方可以将n用二进制表示出来，然后就是若干个x的i次方相加，二进制有1的位置就是x的i次方
        double weigh = x ;
        while (n!=0){
            if(n%2==1){
                res = res*weigh;
            }
            weigh = weigh*weigh;
            n = n/2;
        }
        return res;
        //存在重复计算
        //while(count<=n/2){
        //    temp = temp * temp;
        //    count = count + count;
        //}
        //return temp * myPow(x,n-count);
    }
    public int maxSubArray1(int[] nums) {
        //固定住了不确定的因素（固定右端点），使状态转移变得更容易
        //无后效性：已求解的子问题不受后续阶段的影响，即每个子问题只需求解一次
        //有后效性->无后效性，增大状态数组的维度，题解liweiwei挺好
        int res = nums[0];
        int max=nums[0];//dp[i-1]
        //dp[i] = Math.max(dp[i-1]+nums[i],nums[i])
        for(int i=1;i<nums.length;i++){
            max = Math.max(nums[i],max+nums[i]);
            res = Math.max(res,max);
        }
        return res;
    }
    //分治解法，最大子序列和可以只从左半取，也可以从右半取，也可以跨越取，关键在于跨越取怎么实现
    //方法1：扩散，从mid像两端扩散算出最大值2.维护一个数据结构，因为跨越取等于左半区间中，包含右端点的最大序列和+右半区间中，包含左端点的最大序列和
    public class SubArrayMessage{
        int lSum;int rSum;//包含左，右端点的子序列和，用于计算合并数组的最大连续子序列和（跨越和）
        int subSum;int allSum;//用于计算和并数组的最大左和
        public SubArrayMessage(int lSum, int rSum, int subSum, int allSum) {
            this.lSum = lSum;
            this.rSum = rSum;
            this.subSum = subSum;
            this.allSum = allSum;
        }
    }
    public int maxSubArray(int[] nums){
        return getMessBypartition(0,nums.length-1,nums).subSum;
    }
    //计算当前子区间内的所有信息
    public SubArrayMessage getMessBypartition(int left, int right, int[] nums){
        if(left==right){
            return new SubArrayMessage(nums[left],nums[left],nums[left],nums[left]);
        }
        int mid = (left+right)/2;
        //1.分别计算左右子数组信息
        SubArrayMessage lMess = getMessBypartition(left,mid,nums);
        SubArrayMessage rMess = getMessBypartition(mid+1,right,nums);
        //2.合并计算数组信息
        return mergePartition(lMess,rMess);
    }
    public SubArrayMessage mergePartition(SubArrayMessage lMess,SubArrayMessage rMess){
        int lSum = Math.max(lMess.lSum,lMess.allSum+rMess.lSum);
        int rSum = Math.max(rMess.rSum,rMess.allSum+lMess.rSum);
        int allSum = lMess.allSum+rMess.allSum;
        int subSum = Math.max(Math.max(lMess.subSum,rMess.subSum),lMess.rSum+rMess.lSum);//左和，右和，跨越和三者选最大的
        return new SubArrayMessage(lSum,rSum,subSum,allSum);
    }
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length,n = matrix[0].length;
        List<Integer> res = new ArrayList<>();
        int left = 0,right = n-1;
        int up = 0, down = m-1;
        while (up<=down && left<=right){
            for(int j=left;j<=right;j++){
                res.add(matrix[up][j]);
            }
            up++;
            if(up>down) break;//既然每次都会更新，不妨每个for之后都判断一下，
            // 并且这个if判断是必须的，虽然第二个for会跳过，但是第3个for可以正常运行，这就导致重复，如三行四列或一行两列
            for(int i=up;i<=down;i++){
                res.add(matrix[i][right]);
            }
            right--;
            if(left>right) break;
            for(int j=right;j>=left;j--){
                res.add(matrix[down][j]);
            }
            down--;
            if(down<up) break;
            for(int i =down;i>up;i--){
                res.add(matrix[i][left]);
            }
            left++;
            if(left>right) break;
        }
        return res;
    }
    public boolean canJump(int[] nums) {
        //不能像跳跃游戏2一样通过i==end判断，因为越过去也是能跳到
        //不用想太复杂，很简单的逻辑，读所有点计算maxPos，如果i大于maxPos就说明到不了
        int maxPos =0;
        for(int i=0;i<nums.length;i++){
            if(i>maxPos) return false;//注意：if一定要放前面
            maxPos = Math.max(maxPos,i+nums[i]);
        }
        return true;
    }
    public int[][] merge(int[][] intervals) {
        List<int[]> list = new ArrayList<>();
        Arrays.sort(intervals,(a,b) -> a[0]-b[0]);//还是写不好,或者new Comparator,或Comparator.compareInt(a -> a[0])
        int left = intervals[0][0],right = intervals[0][1];
        for(int i = 1;i<intervals.length;i++){
            //或者for套while(i+1<n&&right>=intervals[i+1][0])
            if(intervals[i][0]<=right){
                right = Math.max(right,intervals[i][1]);
            }else {
                list.add(new int[]{left,right});
                left = intervals[i][0];
                right = intervals[i][1];
            }
        }
        list.add(new int[]{left,right});//别忘
        return list.toArray(new int[][]{});//或new int[0][]
    }
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> res = new ArrayList<>();
        int n = intervals.length;
        int i =0;
        while (i<n && intervals[i][1]<newInterval[0]){//重点，应该是比右边界，比左边界的话，找到插入位置后，需要判断能不能和上一个合并，涉及到i-1，边界处理太复杂
            res.add(intervals[i]);
            i++;
        }
        //需要插到最后
        if(i==n) {
            res.add(newInterval);
            return res.toArray(new int[0][]);
        }
        //找到插入位置,i和intervals一定是能合并的
        int left = newInterval[0],right = newInterval[1];
        left = Math.min(left,intervals[i][0]);
        //看是否需要合并
        while (i<n && intervals[i][0]<=right){
            //可以left = Math.min(left,intervals[i][0]);这样就不需要特判是否是插到最后了
            right = Math.max(intervals[i][1],right);
            i++;
        }
        res.add(new int[]{left,right});
        //添加剩下的元素
        while (i<intervals.length){
            res.add(intervals[i]);
            i++;
        }
        return res.toArray(new int[0][]);
    }
    public int lengthOfLastWord(String s) {
        //也可以读空格，index--
        String s1 = s.trim();
        for(int i=s1.length()-1;i>=0;i--){//别忘了减1，区分减1和最后返回得length
            if(s1.charAt(i)==' '){
                return s1.length() -1 - i;
            }
        }
        return s1.length();
    }
    public int[][] generateMatrix(int n) {
        int[][] matrix = new int[n][n];
        int up=0,down=n-1;
        int left=0,right=n-1;
        int num = 1;
        while (true){//因为是方阵，所以可优化所有判断条件为num<=n*n,然后去掉所有if，区分螺旋1，他不是方阵
            for(int j=left;j<=right;j++){
                matrix[up][j] = num++;
            }
            up++;
            if(up>down) break;
            for(int i=up;i<=down;i++){
                matrix[i][right] = num++;
            }
            right--;
            if(left>right) break;
            for(int j=right;j>=left;j--){
                matrix[down][j] = num++;
            }
            down--;
            if(up>down) break;
            for(int i=down;i>=up;i--){
                matrix[i][left] = num++;
            }
            left++;
            if(left>right) break;
        }
        return matrix;
    }






}
