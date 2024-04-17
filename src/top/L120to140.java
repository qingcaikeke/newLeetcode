package top;

import java.util.*;

//305-308
public class L120to140 {
    public static class TreeNode {
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
    /**
     * 121.买卖股票的最佳时机1：还是没写出来，贪心
     * 122.买卖股票的最佳时机2: 贪心，动态规划
     * 123.买卖股票的最佳时机3: 怎么保证最多完成两笔交易？通过状态的定义
     * 124.二叉树中的最大路径：基本上写出来了，递归
     * 125.验证回文串：双指针+字符串处理
     * 126.
     * 127.
     * 128.最长连续序列：哈希集，注意优化时间
     * 129.求根到叶的数字之和：递归，使用全局变量不返回 或 返回叶的和，然后一层一层向上
     * 130.被围绕的区域:没写出来，本质上就是看懂题，然后用广度或深度遍历
     * 131.分割回文串：动态规划判断回文串+搜索
     * 132.
     * 133.克隆图：图的深度搜索和广度搜索（挺好的一个教学）+map去重防止死循环
     * 134.加油站：散点图解法的思路
     * 135.
     * 136.只出现一次的数字：一个元素出现1次，其他都是两次，异或运算
     * 137.只出现一次的数字2：一个元素出现1次，其他都是三次
     * 138.随机链表的复制：深拷贝，和133克隆图相似，但要注意一些不同于图的性质，可以写的更简化，1.链表可以顺序遍历2.可以优化空间省略哈希表
     * 139.单词拆分:上来想了个搜索还没剪枝，直接超时，这是一道很好的从搜索想到记忆化搜索，再想到动态规划的题
     * 140.
     */

    public int maxProfit1(int[] prices) {
        //1.暴力双循环
        //2.优化，如果后一天比前一天更偏移，一定不会在前一天买
        //理解方式2：把问题转换为要不要在今天卖，如果今天卖的话一定是在今天之前，价格最低的那天买
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for(int i =0;i<prices.length;i++){
            if(prices[i]<minPrice) minPrice = prices[i]; //只有新的最小价格比上一个最小价格还小，在今天买才有可能获得更多利润
            else {
                maxProfit = Math.max(maxProfit,prices[i]-minPrice);
            }
        }
        return maxProfit;
    }
    public int maxProfit20(int[] prices) {
        //求最优化的问题通常要经过一系列步骤，每一步骤有很多选则。
        //贪心算法相比动态规划更简单高效，在每一步都做出在当时看来最佳的选则，即局部最优解，寄希望于这样的选则能达到全局最优
        //他既不看前面（也就是说它不需要从前面的状态转移过来），也不看后面（无后效性，后面的选择不会对前面的选择有影响），
        // 因此贪心算法时间复杂度一般是线性的O(n)，空间复杂度是常数级别的O(1)
        int res = 0 ;
        for(int i=1;i<prices.length;i++){
            res+=Math.max(prices[i]-prices[i-1],0);
        }
        return res;
    }
    public int maxProfit21(int[] prices) {
        //每天有两种选则，要么买卖（操作），要么不操作，这就产生了第一种回溯搜索算法,每天有两种状态（有股票或没股票），根据每天的两种第二天又会有两种（操作，不操作），这就产生了多个分叉
        //然后进行优化，想到动态规划,每层有2^n次方个状态，而用数组**只记录最好的两个**（要么有股票，要么没股票），然后从这两个开始转移，这样就优化成了每层都是（2->4）
        int n = prices.length;
        int[][] dp = new int[n][2];
        dp[0][1] = -prices[0];
        for(int i=1;i<n;i++){
            dp[i][0] = Math.max(dp[i-1][0],dp[i-1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i-1][1],dp[i-1][0] - prices[i]);
        }
        return dp[n-1][0];
        //继续优化空间复杂度：改为四个变量pre0,pre1,cur0,cur1
    }
    public int maxProfit31(int[] prices) {
        //不能用贪心，局部最优解到不了全局最优解（不能选两段上升最快的）
        //要求：必须在再次购买前出售掉之前的股票
        int n = prices.length;
        //初始化第1天的状态，第一次买，第一次卖，第二次买，第二次卖
        int buy1 = -prices[0] , sell1 = 0;
        int buy2 = -prices[0] , sell2 = 0;
        for(int i=1;i<n;i++){
            buy1 = Math.max(buy1,-prices[i]);
            sell1 = Math.max(sell1,buy1+prices[i]);
            buy2 = Math.max(buy2,sell1 - prices[i]);
            sell2 = Math.max(sell2,buy2+prices[i]);
        }
        return sell2;
    }
    public int maxProfit32(int[] prices) {
        int n = prices.length;
        //至少需要2*k+1个变量，第k次买卖+仍未操作，也可以只用2k个，但每次循环需要额外处理buy1
        int[] buy = new int[3];
        int[] sell = new int[3];
        //初始化第0天
        buy[0] = buy[1] = buy[2] = -prices[0];
        for(int i=1;i<n;i++){
            for(int j=1;j<=2;j++){
                buy[j] = Math.max(buy[j],sell[j-1] - prices[i]);
                sell[j] = Math.max(sell[j],buy[j]+prices[i]);
            }
        }
        return sell[2];
    }
    int res = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        dfsMaxPath(root);
        return res;
    }
    public int dfsMaxPath(TreeNode root){
        //走到每一个节点都有三种选则，1.停在节点，左+右+val得到最大值 2.用于给上一层贡献 左+val或 右+val
        //递归适合处理这种规模不同的同一问题
        //从上往下走存在大量重复计算，所以选则从下往上，类似中序遍历
        //记录从下到该节点的最大路径和，然后经过每个节点的最大路径等于节点值+左+右，选大于0的
        if(root==null) return 0;
        int left = Math.max(0,dfsMaxPath(root.left));
        int right = Math.max(0,dfsMaxPath(root.right));
        //如果取通过该节点为结果，左右可以都选
        res = Math.max(res,root.val+left+right);
        //如果往上走的话，左右只能选则一个，否则就不是一条路径了
        return root.val + Math.max(left,right);//上一层要用，返回是什么？自己需要什么，就应该继续给上层什么，自己计算最大路径的时候需要左右的贡献，返沪上一层的就应该是以自己为根的路径能提供的最大贡献
    }
    public boolean isPalindrome(String s) {
        s = s.trim().toLowerCase();
        for (int i=0,j = s.length()-1;i<j;i++,j--){
            while (!Character.isLetterOrDigit(s.charAt(i))) i++;
            while (!Character.isLetterOrDigit(s.charAt(j))) j--;
            if(s.charAt(i)!=s.charAt(j)){
                return false;
            }
        }
        return true;
    }
    //todo 126,127
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int res =0 ;
        for(int num:nums)  set.add(num);
        for(int num:nums){
            if(!set.contains(num+1)){//优化时间，只从一个序列中的最后一个元素开始往前找
                //这样的话确保内层循环中一个数只会被遍历一次，才能达到O(n)
                int count =1;
                while (set.contains(num-1)){
                    count++;
                    num--;
                }
                res = Math.max(res,count);
            }
        }
        return res;
    }
    int ress=0;
    public int sumNumbers(TreeNode root) {
        dfsSumNum(root,0);
        return ress;
    }
    public void dfsSumNum(TreeNode root,int cur){
        //优化：可以void改int，然后下返回上的思想，return dfsSumNum(root.left,cur)+dfsSumNum(root.right,cur);
        if(root==null) return;
        cur = cur*10+root.val;
        if(root.left==null&&root.right==null){
            ress+=cur;
            return;
        }
        dfsSumNum(root.left,cur);
        dfsSumNum(root.right,cur);
    }
    public void solve(char[][] board) {
        //怎么确定一个数是被围绕的 广度搜索？ 没想到逆向思维，找没被环绕的
        //任何边界上的 'O' 都不会被填充为 'X'，任何不在边界上，或不与边界上的'O'相连的'O'最终都会被填充为'X'
        //从边界的所有O开始出发，遍历，把能到的位置置为A，这些就是最终仍为O的
        int m = board.length,n = board[0].length;
        for(int i=0;i<m;i++){
            //可改用广度搜索，用一个队列，因为找到的元素被标记为a了，所以不用重新找
            if(board[i][0]=='O') dfsSolve(m,n,i,0,board);
            if(board[i][n-1]=='O') dfsSolve(m,n,i,n-1,board);
        }
        for(int j=1;j<n-1;j++){
            if(board[0][j]=='O')    dfsSolve(m,n,0,j,board);
            if(board[m-1][j]=='O')   dfsSolve(m,n,m-1,j,board);
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }
    public void dfsSolve(int m,int n,int x,int y,char[][] board){
        if (x < 0 || x == m || y < 0 || y == n || board[x][y] != 'O') {
            return;
        }//边界出发，找所有能连上的，改为A标记
        board[x][y] = 'A';
        dfsSolve(m,n,x-1,y,board);
        dfsSolve(m,n,x+1,y,board);
        dfsSolve(m,n,x,y-1,board);
        dfsSolve(m,n,x,y+1,board);
    }
    public static List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        //用dp保存i到j是否为回文串，否则每次搜索加一个元素就要遍历，复杂度太高
        int n =s.length();
        boolean[][] dp = new boolean[n][n];
        for(int j=0;j<n;j++){//重点在于遍历顺序：1.j一定大于i，所以看上半区，之后判断是该一列一列算还是一行一行算
            //算i，j需要用到i+1，j-1，所以需要先算前一列，所以外层是j，内层是i
            for(int i=0;i<=j;i++){
                if( (i+1>=j-1 || dp[i+1][j-1]) && s.charAt(i)==s.charAt(j)) dp[i][j] = true;
            }
        }
        dfsPartition(res,new LinkedList<>(),0,s,dp);
        return res;
    }
    public static void dfsPartition(List<List<String>> res, List<String> path, int start, String s, boolean[][] dp){
        if(start==s.length()){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i = start;i<s.length();i++){
            //如果start到i是回文串
            if(dp[start][i]){
                path.add(s.substring(start,i+1));
                dfsPartition(res,path,i+1,s,dp);
                path.remove(path.size()-1);
            }
        }
    }
    // todo 132
    static class cloneGraph{
        class Node {
            public int val;
            public List<Node> neighbors;
            public Node() {
                val = 0;
                neighbors = new ArrayList<Node>();
            }
            public Node(int _val) {
                val = _val;
                neighbors = new ArrayList<Node>();
            }
            public Node(int _val, ArrayList<Node> _neighbors) {
                val = _val;
                neighbors = _neighbors;
            }
        }
        Map<Node,Node> map = new HashMap<>();
        public Node cloneGraph1(Node node) {
            //深度/广度优先搜索，怎么解决重复问题？用set，只要一个节点的拷贝已经被new出来了，就把他放到set中
            //搞清楚每层递归做的事，都是先拷贝自己，加入map，然后递归调用去拷贝临接点
            if(node==null) return node;
            if(map.containsKey(node)){
                return map.get(node);
            }
            Node copyNode = new Node(node.val,new ArrayList<>());
            map.put(node,copyNode);
            for(Node n :node.neighbors){
                Node copyN = cloneGraph1(n);
                copyNode.neighbors.add(copyN);
            }
            return copyNode;
        }
        public Node cloneGraph2(Node node) {
            //广度优先搜索如何保证所有的邻接关系?
            //因为广度不会回溯，所以一次遍历的过程要把邻接的所有节点都先拷贝了
            //所以队列中存的元素实际上是，已经拷贝了，但是没有处理邻接关系的
            Queue<Node> queue = new LinkedList<>();
            if(node==null) return node;
            Node copyNode = new Node(node.val,new ArrayList<>());
            map.put(node,copyNode);
            queue.add(node);
            while (!queue.isEmpty()){
                Node poll = queue.poll();
                //拷贝poll出来的节点
                for(Node n : poll.neighbors){
                    if(!map.containsKey(n)){
                        map.put(n,new Node(n.val,new ArrayList<>()));
                        queue.add(n);
                    }
                    //map中不在，意味着需要拷贝，加入队列
                    map.get(poll).neighbors.add(map.get(n));
                }
            }
            return map.get(node);
        }
    }
    public int canCompleteCircuit(int[] gas, int[] cost) {
        //第一想法暴力：优化想到图像，找到最低点，然后如果扫一遍后和大于等于零，就能到，否则返回-1
        //怎么找到最低点  还想到一个问题，到了数组末尾怎么回环回去？可以使用取余i%n
        int min = 0;
        int count=0,start=0;
        for (int i=0;i<gas.length;i++) {
            count = count+gas[i]-cost[i];//算出的是在下一个点的油量
            if(min<count){
                min = count;
                start = (i+1)%gas.length;
            }
        }
        return count>=0? start : -1;
    }
    //todo 135
    public int singleNumber1(int[] nums) {
        //法1，暴力，法2：哈希表 法3：set，出现一次就加入，出现两次就删除
        //两个相同的数异或为0，0与任何数异或为原数
        int res = 0;
        for(int num:nums){
            res = res^num;
        }
        return res;
    }
    public int singleNumber(int[] nums) {
        //要求线性时间，常数空间的话用数电方法，没啥价值，垃圾题
        //法1：开个哈希表储存次数，getOrDefault
        //法2,开个数组记录每个二进制位为1的次数，然后找出所有1次或4次的，复杂度nlog32
        int[] count  = new int[32];
        for(int num:nums){
            for(int i=0;i<32;i++){
                count[i] += num>>i & 1;
            }
        }
        int res = 0;
        for(int i=0;i<32;i++){
            if(count[i]%3==1) res+=1<<i;
        }
        return res;
    }
    static class CopyRanomL {
        static class Node {
            int val;
            Node next;
            Node random;

            public Node(int val) {
                this.val = val;
                this.next = null;
                this.random = null;
            }
        }
        Map<Node,Node> map = new HashMap<>();
        //更好的方法：哈希表加遍历：创建出所有节点的拷贝，再构建指针o(n),O(n)和dfs一样
        //优化空间：不用哈希表，原来是cur-next，第一次遍历后变为cur-copyCur-next,哈希表的作用在于找到拷贝节点
        //而这样修改指向后，可以直接找到拷贝节点，也就是节点的next，然后就可以完成random的指向，然后再拆分成两个链表
        public Node copyRandomList1(Node head) {
            //深度优先搜索，和之前图的拷贝一样
            if(head==null) return null;
            if(map.containsKey(head)){
                return map.get(head);
            }
            Node copyNode = new Node(head.val);
            map.put(head,copyNode);
            copyNode.next = copyRandomList1(head.next);
            copyNode.random = copyRandomList1(head.random);
            return copyNode;
        }
        public Node copyRandomList2(Node head){
            //bfs
            Queue<Node> queue = new LinkedList<>();
            Node copyNode = new Node(head.val);
            queue.add(head);
            map.put(head,copyNode);
            while (!queue.isEmpty()){
                Node poll = queue.poll();
                Node next = poll.next;
                Node random = poll.random;
                if(next==null){
                    map.get(poll).next = null;
                }
                if(next!=null&&!map.containsKey(next)){
                    Node copyNext = new Node(next.val);
                    map.put(next,copyNext);
                    queue.add(next);
                }
                map.get(poll).next = map.get(next);
                if(random==null){
                    map.get(poll).random = random;
                }
                if(random!=null&&!map.containsKey(random)){
                    Node copyRandom = new Node(random.val);
                    map.put(random,copyRandom);
                    queue.add(random);
                }
                map.get(poll).random = map.get(random);
            }
            return copyNode;
        }
    }
    public boolean wordBreak1(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        boolean[] cantFound = new boolean[s.length()];
        return dfsWordBreak(s,set,0,cantFound);
    }
    public boolean dfsWordBreak(String s,Set<String> set,int start,boolean[] cantFound){
        if(start==s.length()){
            return true;
        }
        //剪枝是记录了什么？记录了从s的当前位置之后的字串一定不能被拼接成，如果第二次遇到这个位置就不用往后了
        if(cantFound[start]) return false;//剪枝
        for(int end = start;end<s.length();end++){
            if(set.contains(s.substring(start,end+1))){
                if(dfsWordBreak(s,set,end+1,cantFound)){
                    return true;
                }
            }
        }
        cantFound[start] = true;//剪枝
        return false;
    }
    public boolean wordBreak2(String s, List<String> wordDict){
        //搜索 -》记忆化方法弄掉重复计算，剪枝 -》改变计算顺序，去掉递归，用刷表方式，直接顺序计算
        int n = s.length();
        boolean[] dp = new boolean[n+1];//word可以用多次，所以是一维数组而非二维？
        //搜索：以i开头的字符串是否能被构成，怎么判断的？枚举所有i后面的位置end，看i到end是否再dict中，然后一直往后推（见dfs的for循环）
        //以i结尾的字符串是否能被构成，那就枚举i之前的所有位置
        dp[0] = true;
        for(int i=1;i<=n;i++) {//s的[0,i)能否被构成
            for(String word:wordDict){
                int len = word.length();
                if(i-len>=0){
                    if(s.substring(i-len,i).equals(word) && dp[i-len]){
                        dp[i] = true;
                        break;//注意这个位置，优化为及时退出循环
                    }
                }
            }
        }
        return dp[n];
    }

    //todo 140






}
