package top;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//217-222
public class L060toL080 {
    public void swap(int[] nums, int left, int right) {
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
    }
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
    //217-
    /**
     * 61.旋转链表：成环是为了边界好处理一点，不成环也行，关键在于找到对应位置
     * 62.不同路径:简单动态规划
     * 63.不同路径II:写的结构不好
     * 64.最小路径和:和前面一样，动态规划，简单题
     * 65.
     * 66.加一：模拟，写的不好，思路不够简单
     * 67.二进制求和:模拟，简单题，写的不好
     * 68.
     * 69.x的平方根:二分查找(注意边界值和超界，注意二分条件)，牛顿迭代
     * 70.爬楼梯：动态规划（on），快速次幂(时间复杂度logn)
     * 71.简化路径：写的不好，注意使用什么数据结构，注意边界条件，注意split
     * 72.编辑距离，动态规划，没写出来，当前状态可以由pre1+增转移过来，也可以pre2+删，pre3+改
     * 73.矩阵置0：o（mn）空间->o(m+n)空间->o(1)空间,不难
     * 74.搜索二维矩阵:直接二分或者从右上或左下还是走（类似二叉树）
     * 75.颜色分类，双指针（一次遍历），注意细节，或者桶排序，需要额外空间o（k）和额外时间
     * 76.最小覆盖子串:滑动窗口，hard
     * 77.组合：简单搜索,可以与子集对照
     * 78.子集:多解法 1.两种回溯（for/选或不选）2.选或不选构成二叉树左右节点，使用二叉树遍历 3.枚举二进制，第i位为1就加入path 4.递推，将之前的path复制，一个加入i，一个不加i
     * 79.单词搜索:回溯，不难，细心就行
     * 80.删除有序数组中的重复项 II,引申到元素可以出现k次，通过双指针构造一个窗口大小为k的滑动窗口
     */
    public ListNode rotateRight(ListNode head, int k) {
        if(head==null) return head;
        ListNode cur = head;
        int n = 1;
        while (cur.next!=null){
            cur = cur.next;
            n++;
        }
        if(k%n ==0 )return head;
        ListNode end = cur;
        cur = head;
        for(int i=1;i<=n-k%n;i++){
            cur = cur.next;
        }
        ListNode newHead = cur.next==null? head:cur.next;
        cur.next = null;
        end.next = head;
        return newHead;
    }
    //62
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for(int i=1;i<m;i++){
            dp[0] = 1;
            for(int j=1;j<n;j++){
                dp[j] = dp[j]+dp[j-1];
            }
        }
        return dp[n-1];
    }
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length,n = obstacleGrid[0].length;
        int[] dp = new int[n];
        //初始化第一行
        for(int j= 0;j<n;j++){
            if(obstacleGrid[0][j]==0) dp[j] = 1;
            else break;
        }
        for(int i=1;i<m;i++){
            dp[0] =  dp[0]!=0 && obstacleGrid[i][0]==0 ? 1 :0;//用于初始化第一列，上一行能到达并且当前行没石头
            for(int j=1;j<n;j++){
                if(obstacleGrid[i][j]==0) dp[j] = dp[j]+dp[j-1];//没有石头
                else dp[j] = 0;//else必须有，每行都会更新，不能靠默认的0了
            }
        }
        return dp[n-1];
    }
    public int minPathSum(int[][] grid) {
        int m = grid.length,n = grid[0].length;
        int[] dp = new int[n];
        dp[0] = grid[0][0];
        for(int j=1;j<n;j++){
            dp[j] = dp[j-1]+grid[0][j];
        }
        for(int i=1;i<m;i++){
            dp[0] = dp[0]+grid[i][0];
            for(int j=1;j<n;j++){
                dp[j] = Math.min(dp[j],dp[j-1])+grid[i][j];
            }
        }
        return dp[n-1];
    }
    //todo 65
    public int[] plusOne(int[] digits) {
        int n = digits.length;
        //从后往前找，第一个不为9的位+1,其余置0，如果全是9，新开个数组
        for(int i = n-1;i>=0;i--){
            if(digits[i]!=9){
                digits[i]++;
                return digits;
            }else{
                digits[i] = 0;
            }
        }
        digits = new int[n+1];
        digits[0] = 1;
        return digits;
    }
    //67
    public String addBinary(String a, String b) {
        int m = a.length(),n = b.length();
        int i=m-1,j=n-1,add=0;
        StringBuffer sb = new StringBuffer();
        while (j>=0 || i>=0){
            int ai = i>=0 ? a.charAt(i) - '0' : 0;
            int bi = j>=0 ? b.charAt(i) - '0' : 0;
            int num = ai + bi + add;
            add = num/2;
            sb.append(num%2);
            i--;j--;
        }
        if(add!=0) sb.append("1");
        return sb.reverse().toString();
    }
    //todo 68
    public int mySqrt(int x) {
        int left=0,right = x;//left=1，right=i.max会超界
        int res = 0;
        //不是普通二分，要找到满足mid*mid<=x中最大的mid
        while (left<=right){
            int mid = (left+right)/2;
            if((long)mid*mid<=x){//注意long//也可以mid <= x/mid
                res = mid;
                left = mid+1;
            } else {
                right = mid-1;
            }
        }
        return res;
    }
    //70
    public int climbStairs(int n) {
        //[1,1][1,0]的n次方
        return pow(n);
    }
    public int pow(int n){
        int[][] res= new int[][]{{1,0},{0,1}};
        int[][] p = new int[][]{{1,1},{1,0}};
        while (n!=0){
            //想二进制，除2倒取余 n次方拆成a次方乘b次方乘c次方
            if(n%2==1){
                res = mutiMartix(res,p);
            }
            n = n/2;
            p = mutiMartix(p,p);
        }
        return res[0][0];
    }
    public int[][] mutiMartix(int [][]a,int[][]b){
        int[][] res = new int[2][2];
        for(int i=0;i<2;i++){
            for(int j=0;j<2;j++){
                res[i][j] = a[i][0]*b[0][j]+a[i][1]*b[1][j];
            }
        }
        return res;
    }
    //71
    public String simplifyPath(String path) {
        //字符串处理
        List<String> res = new ArrayList<>();
        String[] strings = path.split("/");//分隔符连续出现或出现在开头会出现空
        for(String s:strings){
            //1:.. 2:. 3:字母
            if(s.equals("..")){//怎么回退到上一级？可以用list,也可以用deque双端队列
                if(!res.isEmpty())res.remove(res.size()-1);
            }
            else if(!s.equals(".") && !s.isEmpty()){//分割后会出现空，我也不到为啥
                res.add(s);
            }
        }
        StringBuffer sb = new StringBuffer();
        if(res.isEmpty()) sb.append("/");//别忘，特殊情况
        for(String s:res){
            sb.append("/");
            sb.append(s);
        }
        return sb.toString();
    }
    //72，隐隐猜到了动态规划，但是不知道怎么转移
    public int minDistance(String word1, String word2) {
        int m = word1.length();int n = word2.length();
        int[][] dp = new int[m+1][n+1]; //1的前i个转移到2的前j个最少需要几步
        //初始化
        for(int j=1;j<=n;j++){
            dp[0][j] = dp[0][j-1]+1;//只能插入
        }
        for(int i=1;i<=m;i++){
            dp[i][0] = dp[i-1][0]+1;//只能删除
        }
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(word1.charAt(i-1)==word2.charAt(j-1)){
                    dp[i][j] = dp[i-1][j-1];
                }else{
                    dp[i][j] = Math.min(Math.min(dp[i][j-1],dp[i-1][j]),dp[i-1][j-1])+1;//增，删，改
                }
            }
        }
        return dp[m][n];
    }
    public void setZeroes(int[][] matrix) {
        //空间优化，o(m+n)空间->o(1)空间，用第一行和第一列取储存某行列是否需要置0
        //o（m+n）需要两个额外的数组bool[m],bool[n].想到用第一行和第一列代替这两个额外的空间
        int m = matrix.length,n = matrix[0].length;
        //1.额外记录第一行和第一列是否需要置0
        boolean row0 = false;
        boolean col0 = false;
        for(int j=0;j<n;j++){
            if(matrix[0][j]==0){
                row0 = true;
                break;
            }
        }
        for(int i=0;i<m;i++){
            if(matrix[i][0]==0){
                col0 = true;
                break;
            }
        }
        //遍历，用第一行和第一列记录
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(matrix[i][j]==0){
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++) {
                if(matrix[0][j]==0 || matrix[i][0] == 0) matrix[i][j] = 0;
            }
        }
        //将第一行置0
        if(row0){
            for(int j=0;j<n;j++){
                matrix[0][j] = 0;
            }
        }
        //将列置0
        if(col0){
            for(int i=0;i<m;i++){
                matrix[i][0] = 0;
            }
        }
    }
    public boolean searchMatrix(int[][] matrix, int target) {
        //从右上或左下走（类似二叉树），从左上走不行，因为需要两个方向一个变大一个变小
        int m = matrix.length,n = matrix[0].length;
        int left=0,right = m*n-1;
        while (left<=right){
            int mid = (left+right)/2;
            int x = mid/n;
            int y = mid%n;
            if(matrix[x][y]==target) return true;
            if(matrix[x][y]<target){
                left = mid+1;
            }else{
                right = mid-1;
            }
        }
        return false;
    }
    //75
    public void sortColors(int[] nums) {
        //法1：桶排序，记录每个元素的数量
        //法2：双指针插入排序，一个记录最后一个0的位置，一个记录第一个2的位置 法3：单指针两次遍历，第一次排号0，第二次排好1
        //记录0和1的话就会麻烦很多
        int index0 = 0,index2 = nums.length-1;
        for(int i=0;i<=index2;i++){
            if(nums[i]==0){
                swap(nums,index0,i);
                index0++;
            }
            if(nums[i]==2){
                swap(nums,index2,i);
                index2--;
                i--;
            }
        }
    }
    //todo 76
    //77
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        dfsCombine(res,new ArrayList<>(),n,1,k);
        return res;
    }
    public void dfsCombine(List<List<Integer>> res,List<Integer> path,int n,int start,int k){
        //直接用k计数，省略一个count
        if(k==0){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=start;i<=n-k+1;i++){//i<=n-k+1
            path.add(i);
            dfsCombine(res,path,n,i+1,k-1);
            path.remove(path.size()-1);
        }
    }
    //78
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        dfsCombine(res,new ArrayList<>(),nums,0);
        return res;
    }
    public void dfsCombine(List<List<Integer>> res,List<Integer> path,int[] nums,int start){
        res.add(new ArrayList<>(path));//每次发生变化都代表产生了一个新的子集
        if(start==nums.length){
            return;
        }
        for(int i =start;i<nums.length;i++){
            path.add(nums[i]);
            dfsCombine(res,path,nums,i+1);
            path.remove(path.size()-1);
        }
    }
    //79
    public boolean exist(char[][] board, String word) {
        boolean[][] visited = new boolean[board.length][board[0].length];
        for(int i=0;i<board.length;i++){
            for(int j=0;j<board[0].length;j++){
                if(dfsExist(visited,board,word,0,i,j)) return true;
            }
        }
        return false;
    }
    public boolean dfsExist(boolean[][] visited,char[][] board, String word,int index,int x,int y){
        //空间复杂度优化，可以不用visited，把board的该位置换为'/0'，然后回溯时还原为s.charAt(index)
        if(index==word.length()){
            return true;
        }
        if(x<0||x>= board.length||y<0||y>= board[0].length
            ||board[x][y] != word.charAt(index)
            ||visited[x][y]){ //别忘了visited
            return false;   //剪枝，这条路不可能和目标字符串匹配成功，立即返回，不需要继续搜索
        }
        visited[x][y] = true;
        //对传入的每个点，搜索其四个方向
            boolean result = dfsExist(visited,board,word,index+1,x-1,y)
            || dfsExist(visited,board,word,index+1,x+1,y)
            || dfsExist(visited,board,word,index+1,x,y-1)
            || dfsExist(visited,board,word,index+1,x,y+1);
            if(result) return true;
        visited[x][y] = false;//回溯visited
        return false;
    }
    //80
    public int removeDuplicates(int[] nums) {
        int behind = 0;
        for(int i=1;i< nums.length;i++){
            if(nums[i]==nums[behind]){
                if(behind!=0 && nums[behind-1]==nums[i]){
                    continue;
                }
            }
            behind++;
            nums[behind] = nums[i];
        }
        return behind+1;
    }
    public int removeDuplicates2(int[] nums) {
        //法2：构造一个窗口大小为2的滑动窗口，slow,slow-1,slow-2
        int behind = 2;//指向最后一个不保留的元素//2->k
        for(int i=2;i< nums.length;i++){//2->k
            //如何判断当前元素i是否应该保留？
            if(nums[i]!=nums[behind-2]){//可认为窗口是behind-1,behind-2,因为数组递增，如果i等于behind-2，i>=behind-1>=behind-2，必有i是第三次出现
                nums[behind] = nums[i];
                behind++;
            }
        }
        return behind;
    }






























}
