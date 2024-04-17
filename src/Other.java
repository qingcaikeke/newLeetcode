import java.util.*;

/**
 * 167.两数之和Ⅱ，输入有序数组
 * 179.最大数，虾皮实习机试，//贪心临项交换法
 * 204.计数质数，埃晒法
 * 209.长度最小的子数组
 * 239.滑动窗口最大值 1.双端队列(单调队列) 2.优先队列,与双端单调队列的区别在于，插入一个新元素，是在队列内部进行了重排序，而没有删除那些应该删除的元素 3.手搓堆排序实现优先队列
 * LCR120.寻找文件副本：原地哈希
 * 268.丢失的数，原地哈希+bool数组
 * 300.最长递增子序列--对照674，最长连续递增子序列
 * 416.分割等和子集
 * 486.预测赢家，递归或动态规划
 * 491.非递减子序列
 * 547.省份数量，显然并查集，感觉深度搜或广度搜也行
 * 628.三个数的最大乘积：排序，数学思想
 * 724.寻找数组的中心下标：前缀和
 * 860.柠檬水找零：简单贪心，或者说就是个模拟
 * 863.二叉树中所有距离为 K 的结点
 * 976.三角形最大周长：简单贪心+排序，枚举最大的变，然后看nums[i-1]+nums[i-2]
 */
public class Other {
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
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    //167.两数之和Ⅱ，输入有序数组
    public int[] twoSum(int[] numbers, int target) {
        //原来是用map加一次扫描，有序想到了二分o(nlogn)和o(1)
        //没想到o(n)的双指针
        int left=0,right=numbers.length-1;
        while (left<right){
            if(numbers[left]+numbers[right]==target) return new int[]{left+1,right+1};
            else if (numbers[left]+numbers[right]>target) {
                right--;
            }else left++;
        }
        return new int[]{-1,-1};
    }
    //179最大
    public String largestNumber(int[] nums) {
        //本质上还是排序，一次可以确定一个元素的位置，如果ab字典序大于ba，那么a应该在b前面\
        //法2：手写快排，以左为基准，比较拼接后的结果，确定一个数的位置，再去递归
        //贪心临项交换法
        String[] sNums = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            sNums[i] = Integer.toString(nums[i]);
        }
        Arrays.sort(sNums,(a,b) -> (b+a).compareTo(a+b));//注意顺序
        if(sNums[0].equals("0")){
            return "0";
        }
        StringBuffer sb = new StringBuffer();
        for (String sNum : sNums) {
            sb.append(sNum);
        }
        return sb.toString();
    }
    //204.计数质数，埃晒法
    public int countPrimes(int n) {
        int count =0;
        boolean[] isNotPrime = new boolean[n];
        for(int i=2;i<n;i++){
            //如果是质数，把他的二倍三倍都标记成非质数
            if(!isNotPrime[i]){
                count++;
                for(int j=i*2;j<n;j+=i){
                    isNotPrime[j] = true;
                }
            }
        }
        return count;
    }
    //209长度最小的子数组 涉及连续子数组：滑动窗口或前缀和
    public int minSubArrayLen1(int target, int[] nums) {
        int res = Integer.MAX_VALUE;
        //前缀和一般不包含自身
        int[] sum = new int[nums.length+1];
        //要求总和大于等于target，数组元素全是正数
        for(int i=1;i<=nums.length;i++){
            sum[i] = sum[i-1]+nums[i-1];
        }
        for(int i=1;i<=nums.length;i++){
            //找第一个大于等于的 if(>=target) right=mid-1 ,else left=mid+1 ,return left
            int need = sum[i] - target;
            //二分去找最后一个小于等于这个need的前缀和，因为前缀和单增，所以可以二分
            //bug1：没考虑找不到的情况
            int left=0,right=i;
            while (left<=right){
                int mid = (left+right)/2;
                if(sum[mid]<=need){//
                    left = mid+1;//
                }else{
                    right = mid-1;
                }
            }
            //相当于return right
            if(right<0) continue;
            res = Math.min(i-right,res);
        }
        return res==Integer.MAX_VALUE? 0 : res;
    }
    public int minSubArrayLen2(int target, int[] nums) {
        int res = Integer.MAX_VALUE;
        int left=0;
        int windowSum =0;
        for(int i =0;i<nums.length;i++){
            windowSum+=nums[i];
            while (windowSum>=target){
                res = Math.min(res,i-left+1);
                windowSum -= nums[left];
                left++;
            }
        }
        return res==Integer.MAX_VALUE? 0 : res;
    }
    //239滑动窗口最大值
    public int[] maxSlidingWindow1(int[] nums, int k) {
        int[] res = new int[nums.length-k+1];
        Deque<Integer> deque = new LinkedList<>();
        for(int i=0;i<nums.length;i++){
            //nums[i]已经进入窗口了，比nums[i]小的元素一定不可能是窗口的最大值
            while (!deque.isEmpty() && deque.peekLast()<nums[i]){
                deque.pollLast();
            }
            deque.addLast(nums[i]);
            //[i-k]已经超出窗口范围了,头需要出去了
            if(i>=k && nums[i-k]==deque.peekFirst()){
                deque.pollFirst();
            }
            if(i>=k-1) res[i-k+1] = deque.peekFirst();
        }
        return res;
    }
    public int[] maxSlidingWindow(int[] nums, int k) {
        //优先级队列，插入一个元素是内部进行了重排序，原来的元素没有被删除，也只能保证队头元素最优先，参考堆
        //PriorityQueue底层就是创建了一个小顶堆
        PriorityQueue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o2[0]-o1[0];//把大的放前面
            }
        });
        for(int i=0;i<k;i++){
            pq.add(new int[]{nums[i],i});
        }
        int[] res = new int[nums.length-k+1];
        res[0] = pq.peek()[0];
        for(int i=k;i<nums.length;i++){
            pq.add(new int[]{nums[i],i});
            while (i-k>= pq.peek()[1]){
                pq.poll();
            }
            res[i-k+1] = pq.peek()[0];
        }
        return res;
    }
    //268：丢失的数
    // 1.排序2.数组哈希3.原地哈希 4.做差法，算1到n所有数的和，再遍历数组求和，做差就是缺的 5.异或，先对0到n做异或，在对数组所有数做异或，就能找到只出现一次的
    public int missingNumber1(int[] nums) {
        boolean[] appear = new boolean[nums.length+1];
        for(int num:nums){
            appear[num] = true;
        }
        for(int i=0;i<nums.length+1;i++){
            if(!appear[i]) return i;
        }
        return -1;
    }
    public int missingNumber2(int[] nums) {
        for(int i=0;i<nums.length;i++){
            if(i!=nums[i]&&nums[i]!=nums.length){
                swap(nums,i,nums[i]);
                i--;
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i) return i;
        }
        return nums.length;
    }
    //LCR120.寻找文件副本：原地哈希
    public int findRepeatDocument(int[] documents) {
        int i =0;
        while (i<documents.length){
            //本身就在正确的位置（哈希后的位置）
            if(documents[i] == i){
                i++;
                continue;
            }
            //本身不在正确的位置，然后发现要换过去的位置已经有一个正确元素了，说明是第二次出现
            if (documents[documents[i]] == documents[i]) {
                return documents[i];
            }
            //换到正确的位置
            swap(documents,i,documents[i]);
        }
        return -1;
    }
    //300.最长递增子序列，没想出来,hot100里的，dp
    //对照674，最长连续递增子序列，贪心
    public int lengthOfLIS(int[] nums) {
        //不改变其余元素的顺序，所以不能用set
        //法1：o(n^2)的动态规划
        int[] dp = new int[nums.length];
        int res = 0;
        Arrays.fill(dp,1);
        for(int i =1;i<nums.length;i++){
            for(int j=i-1;j>=0;j--){
                dp[i] = Math.max(dp[i],dp[j]+1);
            }
            res = Math.max(dp[i],res);
        }
        return res;
    }
    public int findLengthOfLCIS(int[] nums) {
        //双指针贪心，求局部最优，没有递推啥的
        //或者直接扫，来一个len变量，遇到下降的就把len重置为1
        int left=0;int res=0;
        for(int right =1;right<nums.length;right++){
            if(nums[right]<=nums[right-1]){
                left = right;
            }
            res = Math.max(res,right-left+1);
        }
        return res;
    }
    //416分割等和子集
    public boolean canPartition1(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        if(sum%2!=0) return false;
        int target = sum/2;
        //前i个元素能不能凑出j
        boolean[][] dp = new boolean[nums.length][target+1];
        //一定注意行列都要初始化
        //行
        for(int j=1;j<=target;j++){
            if(j==nums[0]) dp[0][j] = true;
        }
        //列
        for(int i = 0;i< nums.length;i++){
            dp[i][0] = true;//重点，看状态转移方程，如果当前看的数正好等于j，那么能凑出来，应该是true，dp[i-1][j-nums[i]];
            //或者直接dp[0][0]为true，然后下面递推的时候记得j要从0开始
        }
        for(int i=1;i<nums.length;i++){
            for(int j=1;j<=target;j++){
                if(j>=nums[i]){
                    dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]];
                }else {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[nums.length-1][target];
    }
    public boolean canPartition2(int[] nums){
        int sum = Arrays.stream(nums).sum();
        if(sum%2!=0) return false;
        int target = sum/2;
        //01背包空间优化
        boolean[] dp = new boolean[target+1];
        //一定注意dp[0]和初始化第一行和遍历i从1开始
        dp[0] = true;
        if (nums[0] <= target) {
            dp[nums[0]] = true;
        }
        for(int i=1;i<nums.length;i++){
            for(int j=target;j>=nums[i];j--){
                dp[j] = dp[j] || dp[j-nums[i]];
            }
        }
        return dp[target];
    }
    public boolean canPartition(int[] nums){
        int sum = Arrays.stream(nums).sum();
        if(sum%2!=0) return false;
        int target = sum/2;
        //搜索，记忆化：记录当前能用这些元素能凑出来多少---就是动态规划
        //想树的深度，可能存在同一深度，目前累加的结果相等的情况，这样再往下进行，二者的子树就会有很多一样的部分，即重复遍历
        int[][] dp = new int[nums.length][target+1];
        for(int i=0;i<nums.length;i++){
            Arrays.fill(dp[i],-1);
        }
        return dfsCan(nums,target,0,dp);
    }
    public boolean dfsCan(int[] nums,int target,int index,int[][] dp){
        if(target==0) return true;
        if(index==nums.length || target<0) return false;
        if(dp[index][target]!=-1){
            //-1代表没见过，1代表从下标index开始选数，能凑出target，0代表凑不出target
            return dp[index][target] == 1;
        }
        //要这个元素或者不要
        boolean b1 = dfsCan(nums, target, index + 1, dp);
        boolean b2 = dfsCan(nums, target - nums[index], index + 1, dp);
        dp[index][target] = (b1 || b2)?  1: 0;
        return b1 || b2;
    }
    //440移掉k位数字
    public String removeKdigits(String num, int k) {
        return null;
    }
    //486 预测赢家，没想出来,递归或动态规划
    public boolean predictTheWinner(int[] nums) {
        //1.递归，每轮有两种选则
        //return dfsPredict(nums,0,nums.length-1)>=0;
        // 2.优化为动态规划
        int n = nums.length;
        int[][] dp = new int[n][n];//剩下元素为[i,j]时，当前玩家能得到的最大收益
        for (int i = 0; i < n; i++) {
            dp[i][i] = nums[i];//dp[i] = nums[i];
        }
        for(int j=1;j<n;j++){//计算ij需要用到下和左//法1：按列从下到上，从左到右-----法2：按行，从下到上，从左到右
            for(int i=j-1;i>=0;i--){
                dp[i][j] = Math.max(nums[i]-dp[i+1][j],nums[j]-dp[i][j-1]);
                //dp[i] = Math.max(nums[i]-dp[i+1],nums[j]-dp[i]);空间复杂度优化，用内层i，j直接省略
            }
        }
        return dp[0][n-1]>=0;//return dp[0]>=0;
    }
    public int dfsPredict(int[] nums,int start,int end){
        //自己回合加分，对方回合减分，最后统计甲得分是否为正
        if(start==end){
            return nums[start];//返回当前回合甲的得分
        }
        int scoreStart = nums[start] - dfsPredict(nums, start + 1, end); //这里第二项从加号改为减号。
        int scoreEnd = nums[end] - dfsPredict(nums, start, end - 1); //这里第二项从加号改为减号。
        return Math.max(scoreStart, scoreEnd); //直接返回最大值就好了，不需要知道当前玩家是谁。
    }
    //非递减子序列,序列中至少有两个元素，可以不相邻
    //重复元素怎么解决
    public static void main(String[] args) {
        int[] nums = new int[]{4,6,7,7,7};
        List<List<Integer>> res = findSubsequences(nums);
        System.out.println(res);
    }
    public static List<List<Integer>> findSubsequences(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        dfsFindSub(res,new ArrayList<>(),0,nums);
        return res;
    }
    public static void dfsFindSub(List<List<Integer>> res,List<Integer> path,int start,int[] nums){
        if(path.size()>1){
            res.add(new ArrayList<>(path));
            //没有return，因为要取树上的节点
        }
        Map<Integer,Integer> map = new HashMap<>();//每层一个新map，用set也行
        for(int i=start;i<nums.length;i++){
            if(!path.isEmpty() && nums[i]<path.get(path.size()-1)){
                continue;
            }
            if(map.getOrDefault(nums[i],0)==1){
                continue;
            }
            path.add(nums[i]);
            map.put(nums[i],1);
            dfsFindSub(res,path,i+1,nums);
            path.remove(path.size()-1);
        }
    }
    //547省份数量，显然并查集，感觉深度搜或广度搜也行
    public int findCircleNum1(int[][] isConnected) {
        //深度搜，一个visited标记已经访问过了，递归
        int n = isConnected.length;
        int res = 0;
        boolean[] isVisited = new boolean[n];
        for(int i=0;i<n;i++){//遍历所有的城市，如果该城市还没有被找过，就深度搜索所有与他相邻的储层是
            if(!isVisited[i]){
                isVisited[i] = true;
                dfsFindCircle(isConnected,i,isVisited);
                res++;
            }
        }
        return res;
    }
    public void dfsFindCircle(int[][] isConnected,int city,boolean[] isVisited){
        //没有显示的搜索出口，出口可以认为是for循环终止
        for(int j = 0;j<isConnected.length;j++){//j从0开始，而非i+1，因为不是按顺序给相邻关系的，可能1和3连，3和2连
            //i先找到第一个j之后，从j开始借着往下找，
            //但i，j之间的还没找过，所以递归调用时j等于新的i（也就是上次的j）+1会出现问题
            if(isConnected[city][j]==1 && !isVisited[j]){
                isVisited[j] = true;
                dfsFindCircle(isConnected,j,isVisited);
            }
        }
    }
    public int findCircleNum2(int[][] isConnected){
        int n = isConnected.length;
        int res = 0;
        boolean[] isVisited = new boolean[n];
        Queue<Integer> queue = new LinkedList<>();
        for(int c=0;c<n;c++){
            if(isVisited[c]) {
                continue;
            }
            queue.add(c);
            while (!queue.isEmpty()){
                int city = queue.poll();//只有没被访问过的才会入队，所以不用判断
                isVisited[city] = true;//把标记放在出队这个位置比较好
                //队里拿出一个元素，找到所有与他相邻并且未访问过的，依次入队，重复进行，直到队列为空
                for(int i=0;i<n;i++){//找所有与city直接相连的
                    if(isConnected[city][i]==1 && !isVisited[i]){
                        queue.add(i);
                    }
                }
            }
            res++;
        }
        return res;
    }
    class UinonFound{
        //查有几个根就能统计有几个集合
        //怎么描述谁是谁的根？int[]
        int[] parent;
        int[] level;//可以加一个层数，保证效率稳定
        public UinonFound(int n){
            parent = new int[n];
            level = new int[n];
            for(int i=0;i<n;i++){
                parent[i] = i;
                level[i] = 0;
            }
        }
        //合并,把x的根改为y
        public void union(int x,int y){
            int xHead = find(x);
            int yHead = find(y);
            //加层数后合并要把低层合并到高层
            if(xHead!=yHead){
                if(level[xHead]<level[yHead]){
                    parent[xHead] = yHead;
                } else if (level[xHead]>level[yHead]) {
                    parent[yHead] = xHead;
                }else {
                    parent[yHead] = xHead;
                    level[xHead]++;
                }
            }
        }
        //查找根:递归,每层做的事都是去找父的父，直到某个节点是根，把他返回，然后倒序逐层将这一支上的所有节点的父都是赋值给他
        public int find(int x){
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }
        //集，判断x和y是否属于同一集合，集需要查找根；合并也需要用到查找跟
        public boolean isConnected(int x, int y) {
            return find(x) == find(y);
        }
    }
    public int findCircleNum(int[][] isConnected){
        int n = isConnected.length;
        UinonFound uf = new UinonFound(n);
        for(int i=0;i<n;i++){
            for(int j=i+1;j<n;j++){
                if(isConnected[i][j]==1) uf.union(i,j);
            }
        }
        int res=0;//有几个根就是有几个集合
        for(int i=0;i<n;i++){
            if(uf.find(i)==i) res++;
        }
        return res;
    }
    //628.三个数的最大乘积：排序，数学思想
    public int maximumProduct(int[] nums) {
        //我的：排序，选3正（最大）或两负（最小）一正（最大）
        //也可以不用排序，通过线性扫描找到max3和min2
        Arrays.sort(nums);
        int n = nums.length;
        return Math.max(nums[0] * nums[1] * nums[n - 1], nums[n - 3] * nums[n - 2] * nums[n - 1]);
    }
    //724寻找数组的中心下标：前缀和
    public int pivotIndex(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        int leftTotal = 0;
        for(int i=0;i<nums.length;i++){
            //nums[i]算两次，既算左和也算右和
            //或者不算nums[i],total-leftTotal-nums[i] = leftTotal,然后只变leftTotal就行
            leftTotal+=nums[i];
            if(leftTotal==sum) return i;
            sum-=nums[i];
        }
        return -1;
    }
    //863.二叉树中所有距离为 K 的结点,先建图，然后bfs
    //建图有两种方式，一种邻接表，一种用map储存每个节点的父节点
    class DistanceK{
        Map<Integer,TreeNode> parents = new HashMap<>();
        List<Integer> res = new ArrayList<Integer>();
        public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
            findParent(root);
            findKdis(target,null,0,k);
            return res;
        }
        //构建父节点,题里给了节点值不重复,遍历二叉树的同时填map
        public void findParent(TreeNode root){
            if(root==null) return;
            if(root.left!=null) {
                parents.put(root.left.val,root);
                findParent(root.left);
            }
            if(root.right!=null){
                parents.put(root.right.val,root);
                findParent(root.right);
            }
        }
        //深度搜，注意去重，from起到了去重的作用，因为是二叉树，所以不需要set
        public void findKdis(TreeNode node,TreeNode from,int depth,int k){
            if(node==null) return;
            if(depth==k){
                res.add(node.val);
                return;
            }
            if(node.left!=from) findKdis(node.left,node,depth+1,k);
            if(node.right!=from) findKdis(node.right,node,depth+1,k);
            if(parents.get(node.val)!=from) findKdis(parents.get(node.val),node,depth+1,k);
        }
    }

}
