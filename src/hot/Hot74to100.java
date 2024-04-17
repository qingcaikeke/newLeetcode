package hot;

import java.util.*;

public class Hot74to100 {
    public static class TreeNode {
        int val;
        Hot51to100.TreeNode left;
        Hot51to100.TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, Hot51to100.TreeNode left, Hot51to100.TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    public static class ListNode {
        int val;
        Hot51to100.ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, Hot51to100.ListNode next) {
            this.val = val;
            this.next = next;
        }
    }
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    /**
     * 堆
     * 74.数组中的第K个最大元素,要求o(n)时间 1.基于快排 2. 基于堆 3. 基于桶排序
     * 75.前 K 个高频元素：1.基于快排 2. 基于堆 3.基于桶排序，比上一个多了如何处理频率和元素的映射
     * 76.数据流的中位数：大小堆一起维护中位数
     * 贪心
     * 77.买卖股票最佳时机：记录当天之前的最小值，算当天最大利润
     * 78.跳跃游戏：记录i之前能到达的最大位置，如果i超过了最大位置，那么false
     * 79.跳跃游戏II：坑很多，一直错
     * 80.划分字母区间：和跳跃游戏差不多
     * 动态规划
     * 81.爬楼梯，必须秒，记得矩阵算法
     * 82.杨辉三角：秒
     * 83.打家劫舍:秒
     * 84.完全平方数：没思路，动态规划要用来做什么;法2：广度优先搜索也没想到（实际上应该叫层序遍历）
     * 85.零钱兑换：好题：搜索到记忆搜索到dp 完全背包？
     * 86.单词拆分：好题：搜索到记忆搜索到dp
     * 87.最长递增子序列：没思路，动态规划o(n^2)，双层for的dp和二维dp？还有个优化到o(nlogn)的解法
     * 88.乘积最大子数组:dp，没啥要注意的点
     * 89.分割等和子集:应该算是个二维dp？这种的写不好，想不到,本质是个01背包问题
     * 90.最长有效括号：没想到动态规划，子问题没定义出来；栈也有点反人类
     * 多维动态规划
     * 91.不同路径：从左或上转移过来，二维dp秒了，优化直接写
     * 92.最小路径和：和上面的一样，秒了
     * 93.最长回文子串：没啥压力，三角dp
     * 94.最长公共子序列：做出来了，典型二维动态规划题，二维转一维稍微复杂些
     * 95.编辑距离：没啥压力，方形dp
     * 技巧
     * 96.只出现一次的数字：初始化为0，两数异或，一个数出现两次会消为0
     * 97.多数元素：阵营法，挺有意思
     * 98.颜色分类：就是个双指针
     * 99.下一个排列:思路对，有小错误
     * 100.寻找重复数
     */
    class FindKthLargest1{
        public int findKthLargest(int[] nums, int k) {
            //基于快排，快排每次可以返回一个元素的正确位置，看是不是要的第k个，不是根据大小情况判断去左半面还是去右半面，这样每次只需要一般
            return quickSort(nums,0,nums.length-1,k);
        }
        public int quickSort(int[] nums,int left,int right,int k){
            //可以不用原地快排，建三个list，分别储存大于小于等于，之后决定去哪个list里搜
            //private int quickSelect(List<Integer> nums, int k) {}
            int index = partion(nums,left,right);
            if(index==nums.length-k){
                return nums[index];
            }
            if(index<nums.length-k){
                return quickSort(nums,index+1,right,k);
            }
            else {
                return quickSort(nums,left,index-1,k);
            }
        }
        public int partion(int[] nums,int left,int right){
            int mid = (left+right)/2;
            int smaller = left-1;
            int pivot = right;
            //每个元素和最后一个比，比他大就向后，比他小就和前面指向最小的指针交换(小于还是小于等于？)，然后继续向后
            for(int i=left;i<=right;i++){
                if(nums[i]<=nums[pivot]){
                    smaller++;
                    swap(nums,i,smaller);
                }
            }
            return smaller;
        }
    }
    public int findKthLargest(int[] nums, int k) {
        //直接堆排序复杂度nlogn，本题需要建一个大小为k的小顶堆，然后遇到大的就进堆，最后剩下的就是k个最大的
        //这样复杂度为：建堆：o(k)+若干次调整：(n-k)log(k) ————近似o(n)
        //取前k个建堆
        for(int i=k/2-1;i>=0;i--){//起始位置别弄错
            //无序数组从下（最后一个非叶节点）往上建堆复杂度才是o(n)
            heapify(nums,i,k);
        }
        //遇到大的就进堆，最后堆里一定存的是topK
        for(int i = k;i<nums.length;i++){
            //堆顶是nums[0]
            if(nums[i] > nums[0]){//等于进不进堆？不影响
                swap(nums,i,0);
                heapify(nums,0,k);
            }
        }
        return nums[0];
    }
    public void heapify(int[] nums,int i,int heapSize){
        int left = 2*i+1;int right = 2*i+2;int min = i;
        //构造小根堆
        if(left<heapSize && nums[left]<nums[min]){
            min = left;
        }
        if(right<heapSize && nums[right]<nums[min]){
            min = right;
        }
        if(min!=i){
            swap(nums,i,min);
            heapify(nums,min,heapSize);
        }
    }
    public int[] topKFrequent1(int[] nums, int k) {
        //map统计频次，频次建堆，怎么根据频次反向得到数据？再来一个map？
        //解决方法：堆里存数字，不过堆化比较的是map.get
        Map<Integer,Integer> map  = new HashMap<>();
        for(int num:nums){
            map.put(num,map.getOrDefault(num,0)+1);
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>( (a,b) -> map.get(a)-map.get(b));
        for(int num : map.keySet()){
            if(pq.size()<k){
                pq.add(num);
            }else if(map.get(num)>map.get(pq.peek())){
                pq.poll();
                pq.add(num);
            }
        }
        int[] res = new int[pq.size()];//list转array不好转，需要List.stream().mapToInt(Integer::intValue).toArray();
        for (int i = 0; i < res.length; i++) {
            res[i] = pq.poll();
        }
        return res;
    }
    class topKFrequent2{
        public int[] topKFrequent(int[] nums, int k){
            Map<Integer,Integer> map  = new HashMap<>();
            for(int num:nums){
                map.put(num,map.getOrDefault(num,0)+1);
            }
            //如何把频率反向映射回元素？list<int[]> int[0]是num，int[1]是count，然后可以根据索引索引list
            List<int[]> list = new ArrayList<>();
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                list.add(new int[]{entry.getKey(),entry.getValue()});//数组{}里给出初始化，就不要再给长度
            }
            int[] res = new int[k];
            quickSort(list,0,list.size()-1,k,res,0,k);
            return res;
        }
        public void quickSort(List<int[]> list,int start,int end,int k,int[] res,int resIndex,int len){
            if(resIndex==len) return;
            int index = partiton(list, start, end);
            if(end-index >= k){//全在后半部分,index也不要
                quickSort(list,index+1,end,k,res,resIndex,len);
            }else{
                //后面的全要还不够，还要一部分前面的
                for(int i=index;i<=end;i++){
                    res[resIndex++] = list.get(i)[0];
                }
                //if(k>(end-index+1))
                quickSort(list,start,index-1,k-(end-index+1),res,resIndex,len);
            }
        }
        public int partiton(List<int[]> list,int start,int end){
            int pivot = end; //1.基准
            int smaller = start-1; //2.-1
            for(int i=start;i<=end;i++){
                if(list.get(i)[1]<=list.get(pivot)[1]){ //3.小于等于基准
                    smaller++;//4.先加
                    Collections.swap(list,smaller,i);
                }
            }
            return smaller; //5.返回smaller
        }
    }
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        List<Integer>[] list = new List[nums.length+1];
        //以次数为对应的容器下标存到容器中，值存放数据，这样空间耗得不大
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int index = entry.getValue();
            if (list[index] == null) {
                list[index] = new ArrayList<>();
            }
            list[index].add(entry.getKey());
        }
        List<Integer> res = new ArrayList();
        for(int i=nums.length; i>=0&&res.size()<k; i--){
            if(list[i]!=null){
                res.addAll(list[i]);
            }
        }
        return res.stream().mapToInt(Integer::intValue).toArray();
    }
    class MedianFinder {
        //左边大顶堆，右边小顶堆，小的加左边，大的加右边，平衡俩堆数，新加就弹出，堆顶给对家，奇数取多的，偶数取除2
        PriorityQueue<Integer> queueMin;
        PriorityQueue<Integer> queueMax;
        public MedianFinder() {
            queueMin = new PriorityQueue<>((a,b) -> b-a);//大顶堆
            queueMax = new PriorityQueue<>((a,b) -> a-b);//小顶堆
        }

        public void addNum(int num) {
            //记得要维持小于中位数的堆的数目 不超过 大于的1
            if(queueMin.isEmpty() || num<queueMin.peek()) {
                queueMin.add(num);
                if(queueMin.size()>queueMax.size()+1){
                    queueMax.add(queueMin.poll());
                }
            }else{
                queueMax.add(num);
                if(queueMin.size()<queueMax.size()){
                    queueMin.add(queueMax.poll());
                }
            }
        }

        public double findMedian() {
            if(queueMin.size()>queueMax.size()){
                return queueMin.peek();
            }
            return (queueMin.peek()+queueMax.peek())/2.0;
        }
    }
    public int maxProfit(int[] prices) {
        //记录之前的最小值
        int preMin = Integer.MAX_VALUE;
        int res=0;
        for(int i =0;i<prices.length;i++){
            preMin = Math.min(preMin,prices[i]);
            res = Math.max(prices[i]-preMin,res);
        }
        return res;
    }
    public boolean canJump(int[] nums) {
        int maxPos = 0;
        for(int i=0;i<nums.length;i++){
            if(i>maxPos){
                return false;
            }
            maxPos = Math.max(maxPos,i+nums[i]);
        }
        return true;
    }
    public int jump(int[] nums) {
        //不是每次更新最远距离的时候就要步数+1,而是到了最远距离再加
        //因为比方说第一次可以调到7，然后发现在3的时候可以跳到9，4的时候可以跳到10，那么跳到10应该需要两步而不是三步
        //只有一个元素的时候应该返回0
        int maxPos=0; int res=0;int end=0;//需要三个变量而非两个，注意end的作用
        for(int i=0;i<nums.length;i++){
            //res次最远能到end
            maxPos = Math.max(maxPos,i+nums[i]);
            if(maxPos>= nums.length-1){
                if(end==nums.length-1) return res;
                else  return res+1;
            }
            if(i==end){
                res++;
                end = maxPos;
            }
        }
        return res;
    }
    public List<Integer> partitionLabels(String s) {
        //每个字母最多出现在一个片段中
        List<Integer> res = new ArrayList<>();
        //放入一个元素，更新end，直到i==end得到第一个分片
        Map<Character,Integer> map = new HashMap<>();
        //记录每一个字母的最后一个下标
        for(int i =0;i<s.length();i++){
            map.put(s.charAt(i),i);
        }
        int start=0,end=0;
        for(int i=0;i<s.length();i++){
            end = Math.max(end,map.get(s.charAt(i)));
            if(i==end){
                res.add(i-start+1);
                start = i+1;
            }
        }
        return res;
    }
    public int rob(int[] nums) {
        if(nums.length==1) return nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0],nums[1]);
        for(int i=2;i<nums.length;i++){
            dp[i] = Math.max(dp[i-1],dp[i-2]+nums[i]);
        }
        return dp[nums.length-1];
    }
    public int numSquares1(int n) {
        int[] dp = new int[n+1];//需要多少个平方数能构成i
        for(int i=1;i<=n;i++){
            dp[i] = i;
            for(int j=1;j*j<=i;j++){
                dp[i] = Math.min(dp[i],dp[i-j*j]+1);
            }
        }
        return dp[n];
    }
    public int numSquares2(int n) {
        //广度搜存在大量重复计算，动态规划可以看作是广度搜的优化
        Queue<Integer> queue = new LinkedList<>();
        queue.add(n);
        int level = 0;//记录第几层
        while (!queue.isEmpty()){
            level++;
            int size = queue.size();
            for(int i=0;i<size;i++){
                int num = queue.poll();
                for(int j=(int)Math.sqrt(num);j>=1;j--){//从后往前会快很多
                    if(j*j==num){
                        return level;
                    }
                    queue.add(num-j*j);
                }
            }
        }
        return level;
    }
    public int coinChange1(int[] coins, int amount) {
        //搜索 --- 记忆化搜索 ---- 动态规划  也可以先给硬币排序再搜索结合剪枝优化
        //记忆化搜索是从上到下，从amount开始，动态规划是从下到上，从1开始
        //凑不出来要返回-1，怎么实现
        int[] dp = new int[amount+1];
        for(int i=1;i<=amount;i++){
            int min = Integer.MAX_VALUE-1; //其实赋值成amount就行，因为硬币面值最小一块
            for(int coin:coins){
                if(i>=coin) min = Math.min(min,dp[i-coin]);
            }
            dp[i] = min+1;
        }
        return dp[amount]==Integer.MAX_VALUE? -1 :dp[amount];
    }
    public int coinChange2(int[] coins, int amount) {
        int[] visited = new int[amount];//已经计算过所需最小硬币数的
        return dfsCoin(coins,amount,visited);
    }
    public int dfsCoin(int[] coins,int amount,int[] visited){
        if(amount<0) return -1;
        if(amount==0) return 0;
        if(visited[amount-1]!=0) return visited[amount-1];
        int min = Integer.MAX_VALUE;
        for(int coin:coins){
            int res = dfsCoin(coins, amount - coin, visited);
            if(res>=0) min = Math.min(min,res+1);
        }
        visited[amount-1] = (min==Integer.MAX_VALUE? -1 : min);
        return visited[amount-1];
    }
    public boolean wordBreak(String s, List<String> wordDict) {
        //搜索（bfs，dfs都行）不剪枝（从start+1枚举end的位置，set判断字串和word是否相同）--搜索剪枝（记录从start开始不能达到）--动态规划
        boolean[] dp = new boolean[s.length()+1];//s的[0,i)能否被构成
        dp[0] = true;
        for(int i=1;i<=s.length();i++){
            for(String word:wordDict){
                int len = word.length();
                if(i-len>=0 && s.substring(i-len,i).equals(word) && dp[i-len]) { //equals改set就和搜索的思路一样了，除了计算顺序
                    dp[i] = true;
                    break;//别忘
                }
            }
        }
        return dp[s.length()];
    }
    public int lengthOfLIS(int[] nums) {
        if(nums.length==0) return 0;
        int[] dp = new int[nums.length];
        int res=1;
        for(int i=0;i<nums.length;i++){
            dp[i] = 1;
            for(int j=0;j<i;j++){
                if(nums[j]<nums[i]) dp[i] = Math.max(dp[i],dp[j]+1);
            }
            res = Math.max(dp[i],res);
        }
        return res;
    }
    public int maxProduct(int[] nums) {
        int n = nums.length;
        int res=nums[0];//不要随便赋一个0之类的
        int[] maxDp = new int[n];
        int[] minDp = new int[n];
        maxDp[0] = nums[0];minDp[0] = nums[0];
        for(int i=1;i<n;i++){
            maxDp[i] = Math.max(nums[i],Math.max(maxDp[i-1]*nums[i],minDp[i-1]*nums[i]));
            minDp[i] = Math.min(nums[i],Math.min(maxDp[i-1]*nums[i],minDp[i-1]*nums[i]));
            res = Math.max(res,maxDp[i]);
        }
        return res;
    }
    public boolean canPartition1(int[] nums) {
        int sum=0;
        for(int num:nums){
            sum+=num;
        }
        if((sum&1)==1) return false;
        int target = sum/2;
        int[][] dp = new int[nums.length][target+1];
        for(int i=0;i<nums.length;i++){
            Arrays.fill(dp[i],-1);
        }
        return dfsCanp(nums,target,0,dp);
    }
    public boolean dfsCanp(int[] nums,int target,int index,int[][]dp){
        if(target<0) return false;
        if(index==nums.length){
            return false;
        }
        if(dp[index][target]!=-1){
            return dp[index][target] == 1;
        }
        if(nums[index]==target) {
            dp[index][target] = 1;
            return true;
        }
        boolean p1 = dfsCanp(nums, target, index + 1, dp);
        boolean p2 = dfsCanp(nums, target - nums[index], index + 1, dp);
        dp[index][target] = (p1 || p2) ? 1: 0;
        return p1||p2;
    }
    public boolean canPartition2(int[] nums) {
        //搜索：从0到n这些数能否凑出target，用一个index索引，两种方法，用它或不用他
        //记忆化搜搜：dp储存从index能否凑出target，注意dp有三种状态，没看过，看过凑不出，看过凑得出
        //搜索找到所有子集，看和是否为sum/2
        //等价于0-1背包问题，容量为target的背包最多能装多少石头，再看最大值是否为target
        int sum=0;
        for(int num:nums){
            sum+=num;
        }
        if((sum&1)==1) return false;
        int target = sum/2;
        boolean[][] dp = new boolean[nums.length][target+1];
        //初始化第一行
        dp[0][0] = true; dp[0][nums[0]] = true;
        for(int i=1;i<nums.length;i++){ //数组中的前i+1个数能否凑出j-------外层是宝石，区分完全背包
            dp[i][0] = true;
            for(int j=0;j<=target;j++){
                if(j>=nums[i]) dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]];
                else dp[i][j] = dp[i-1][j];
            }
        }
        return dp[nums.length-1][target];
    }
    public int longestValidParentheses1(String s) {
        //核心一定要想清楚，计算当前问题时，假设子问题全计算出来了已经，看能否推出当前问题
        int[] dp = new int[s.length()];//以i结尾的有效括号长度
        int res=0;
        for(int i=1;i<s.length();i++){
            if(s.charAt(i)=='(') continue;
            //2.当前是右括号  2.1前一个是左括号  2.2前一个是右括号
            if(s.charAt(i-1)=='('){
                dp[i] = (i>=2 ? dp[i-2] : 0) + 2;
            } else if (s.charAt(i-1)==')') {  //类似 ()(())
                if(i-1-dp[i-1]>=0 && s.charAt(i-1-dp[i-1])=='('){
                    dp[i] = dp[i-1] + 2 + (i-2-dp[i-1]>=0 ? dp[i-1-dp[i-1]-1] : 0);
                }
            }
            res = Math.max(res,dp[i]);
        }
        return res;
    }
    public int longestValidParentheses2(String s){
        //栈里存下标，左括号入栈，右括号出栈
        //如果栈为空，说明右括号没能被匹配，下标入栈，表示最后一个没右被匹配的括号
        //每次执行完后，当前元素下标减栈顶元素下标就是一组有效的括号长度
        Stack<Integer> stack = new Stack<>();
        int res=0;
        stack.push(-1);//解决边界条件,保证当有左括号出栈时，有栈顶元素可以配合计算长度
        for(int i=0;i<s.length();i++){
            if(s.charAt(i)=='('){
                stack.push(i);
            }else {
                stack.pop();
                if(stack.isEmpty()){
                    stack.push(i);
                }else{
                    int len = i-stack.peek();
                    res = Math.max(res,len);
                }
            }
        }
        return res;
    }
    public String longestPalindrome(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        int start=-1,longest = 0;
        for(int j=0;j<n;j++){
            for(int i=0;i<=j;i++){
                if(s.charAt(i)==s.charAt(j) && (i>=j-2 || dp[i+1][j-1])) {
                    if(j-i+1>longest){
                        start = i;
                        longest = j-i+1;
                    }
                    dp[i][j] = true;
                }
            }
        }
        return s.substring(start,start+longest);
    }
    public int longestCommonSubsequence1(String text1, String text2) {
        //搜索 -- 二维动态规划 ------ 两个一维
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m+1][n+1];//1的0,i结尾和2的0,j的字符的最长子序列
        //不用初始化，就都是0
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(text1.charAt(i-1)==text2.charAt(j-1)) dp[i][j] = dp[i-1][j-1]+1;
                else dp[i][j] = Math.max(dp[i-1][j],dp[i][j-1]);//不用三个比，因为这两个一定都大于等于dp[i-1][j-1]
            }
        }
        return dp[m][n];
    }
    public int longestCommonSubsequence2(String text1, String text2) {
        //空间优化：1个数组
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[2][n+1];
        for(int i=1;i<=m;i++) {
            for (int j = 1; j <= n; j++) {
                if(text1.charAt(i-1)==text2.charAt(j-1)) dp[i%2][j] = dp[(i-1)%2][j-1]+1;
                else dp[i%2][j] = Math.max(dp[(i-1)%2][j],dp[i%2][j-1]);
            }
        }
        return dp[m%2][n];
    }
    public int longestCommonSubsequence3(String text1, String text2) {
        //空间优化：2个数组(通过取余的方式，实现滚动数组)
        int m = text1.length();
        int n = text2.length();
        int[] dp = new int[n+1];
        for(int i=1;i<=m;i++){
            int upLeft = 0;//每次进入新的一行都是0，可以理解为没优化的dp[i][0]
            for(int j=1;j<=n;j++){
                int temp = dp[j];//本次会更新dp[j]，但是算下一列的时候，会用到原来的dp[j]，而他们存在同一个位置
                if(text1.charAt(i-1)==text2.charAt(j-1)) dp[j] = upLeft+1;//用原来的j-1
                else dp[j] = Math.max(dp[j-1],dp[j]);//用新算出来的j-1
                upLeft = temp;
            }
        }
        return dp[n];
    }
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m+1][n+1]; //储存1的前j转换为2的前i个所需要的最小操作数
        //初始化
        dp[0][0] = 0;
        for(int i=1;i<=m;i++){
            dp[i][0] = i;//1取i个凑2的0个，只能删除
        }
        for(int j=1;j<=n;j++){
            dp[0][j] = j;//1取0个凑2的j个，只能增加
        }
        //计算顺序 -- 从左往右，从上往下？从上往下，从左往右？--都行,但这种方形区的就一行一行来吧，不用像那种三角的
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(word1.charAt(i-1)==word2.charAt(j-1)) dp[i][j] = dp[i-1][j-1];//别忘
                //删除一个,替换一个,增加一个
                else dp[i][j] = Math.min(dp[i-1][j] ,Math.min(dp[i-1][j-1] ,dp[i][j-1]))+1;
            }
        }
        return dp[m][n];
    }
    public int majorityElement(int[] nums) {
        //法1：哈希表 法2：排序，索引o(n/2)的元素一定是众数
        // 怎么做到时间o(n),空间o(1)
        //如果一个数组有大于一半的数相同，那么任意删去两个不同的数字，新数组还是会有相同的性质。
        //可以先随便来一个人占领阵营，如果是同样的数，阵营人数加一；否则减一；如果阵营人数为0，那么新来的人占领阵营，最后剩下的一定是众数
        int count =1;
        int candidate = nums[0];
        for(int i=1;i<nums.length;i++){
            if(nums[i]==candidate) count++;
            else {
                if(count==0){
                    candidate = nums[i];
                    count++;
                }
                else{
                    count--;
                }
            }
        }
        return candidate;
    }
    public void sortColors(int[] nums) {
        int left = 0,right = nums.length-1;
        for(int i=0;i<=right;i++){
            if(nums[i]==0){
                swap(nums,i,left);
                left++;
                //i--;这块一定不能有i--，[2,0,2,1,1,0]，如果i和left指向同一个位置，循环后left动了，i没动，left跑到i前面去了
                //为什么不用i--，i扫过之后把2都挪到后面了，i的前面只可能是0或1，left指向第一个1的位置，要么i和left在同一位置，交换也是换过来个1，直接往后走就行
            } else if (nums[i] == 2) {
                swap(nums,i,right);
                right--;
                i--;//换过来这个元素也要观察
            }
        }
        return;
    }
    public void nextPermutation(int[] nums) {
        //从后往前，找第一个升序段的位置i，i+1，从i+1到n-1，找比i大的中最小的那个，交换，剩下元素升序排列  先排列？
        for(int i=nums.length-2;i>=0;i--){
            if(nums[i]<nums[i+1]){//找升序段,此时升序段后必定一直降序
                int j = i+1;
                while (j<nums.length && nums[i]<nums[j]){
                    j++;
                }
                swap(nums,i,j-1);
                Arrays.sort(nums,i+1,nums.length);
                return;
            }
        }
        Arrays.sort(nums);//完全降序的话就是最大，返回升序
    }
    public int findDuplicate1(int[] nums) {
        //要求空间o(1)否则直接set去重就行了
        //二分查找：思路来源于，n+1个数放到n个抽屉，一定有一个抽屉放了两个
        //先猜一个数mid，统计小于等于mid的数的个数，如果个数大于mid，那么重复的数字一定属于left到mid
        //是在 [1..n]查找 nums 中重复的元素，而非在nums中找
        //二分的题，当你找到要对「什么」进行二分的时候，就已经成功一半了。
        int left=1 ,right = nums.length-1;
        while (left<right){//有无等于，有的化最后剩余一个元素进入循环，一定是解，然后会死循环
            //一般情况下，小于等于是在循环体中直接查找元素，小于是在循环体内部排除元素
            int mid = (left+right)/2;
            int count=0;
            for(int num:nums){
                if(num<=mid) count++;
            }
            if(count>mid){
                right = mid;
            }else {
                left = mid+1;
            }
        }
        return right;
    }
    public int findDuplicate(int[] nums) {
        //建立映射关系，根据n去访问fn，像链表一样，重复元素在环的入口位置
        int slow = 0,fast=0;
        slow = nums[0]; fast = nums[nums[0]];//先走一步，否则一直进不了while
        while (slow!=fast){
            slow = nums[slow];     //slow = slow.next;
            fast = nums[nums[fast]];    //fast = fast.next.next;
        }
        int i1 = 0 ,i2 = slow;
        while (i1!=i2){
            i1 = nums[i1];
            i2 = nums[i2];
        }
        return i1;
    }




}
