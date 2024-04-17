package hot;

import java.util.*;

public class Hot51to100 {
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
    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    /**
     * 图论
     * 51.岛屿数量，dfs或bfs，判断是否访问过，要么visited，要么修改原值
     * 52.腐烂的橘子：多源广度优先搜索,广度搜不熟，本题可以看作一个最小路径问题，最小路径用bfs
     * 53.课程表：拓扑排序，bfs（出入度，更重要）或dfs（三种状态）
     * 54.实现前缀树:多叉树
     * 回溯
     * 55.全排列（对照子集）：借助used数组
     * 56.子集：借助start变量
     * 57.电话号码的数字组合:秒了
     * 58.组合总和：start变量
     * 59.括号生成：记录左右个数
     * 60.单词搜索：图的遍历，visited数组防止重复遍历
     * 61.分割回文串：start
     * 62.N皇后:基于set回溯
     * 二分查找
     * 63.搜索插入位置：理解好题意：本质上就是找第一个大于等于target的，然后就是细节，if怎么写，返回值怎么写
     * 64.搜索二维矩阵：二维二分，z字形递增，本质上是一个先展开成一维再还原回二维坐标的过程
     * 65.在排序数组中查找元素的第一个和最后一个位置：两次二分，找第一个大于等于target的和最后一个小于等于target的
     * 66.搜索旋转排序数组：主要还是细节
     * 67.寻找旋转排序数组中的最小值
     * 68.寻找两个正序数组的中位数，想的不够准确，实现也没有实现出来，把本题当成找两个有序数组的第k个数
     * 栈
     * 69.有效的括号
     * 70.最小栈:做了很多次了
     * 71.字符串解码：栈的思想，类似逆波兰表达式
     * 72.每日温度:单调栈，找右侧比当前元素大的第一个元素
     * 73.柱状图中最大的矩形：先想枚举，暴力解法必须会
     */
    public int numIslands(char[][] grid) {
        int m = grid.length,n = grid[0].length;
        int res=0;
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]=='1'){//是陆地
                    res++;
                    bfsLands(grid,i,j);
                }
            }
        }
        return res;
    }
    private void bfsLands(char[][] grid,int x,int y){
        int[] dx = new int[]{1,-1,0,0};
        int[] dy = new int[]{0,0,1,-1};
        int m = grid.length,n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{x,y});
        while (!queue.isEmpty()){
            int size = queue.size();//没必要，只要搜一下就行，不用记录是第几层,直接符合要求就把四个方向塞进去，然后取的时候判断是否要塞下一层
            int[] poll = queue.poll();
            x = poll[0];y = poll[1];
            if(x>=0&&x<m && y>=0&&y<n && grid[x][y]=='1'){
                grid[x][y] = '0';
                for(int j=0;j<4;j++){
                    int nx = poll[0]+dx[j];
                    int ny = poll[1]+dy[j];
                    queue.add(new int[]{nx,ny});
                }
            }
        }
    }
    public int orangesRotting(int[][] grid) {
        //1.从所有腐烂的地方同时广度搜一圈，然后再一圈，问题：什么时候结束？问题：怎么判断无法全部腐烂
        //广度搜借助队列，把下一层的元素全放入队列，然后修改本层的元素，这样就不会出现重复入队的情况，队空说明能搜到的范围全搜过了
        //判断全部腐烂：1.搜过的值被修改，遍历找还有没有没腐烂的 2.计数所有没腐烂的，腐化一个减1，看最后是不是0
        Queue<int[]> queue = new LinkedList<>();
        int m = grid.length,n = grid[0].length;
        int count=0;//初始时好橘子总数
        int res=0;//感染次数
        //初始化，烂橘子入queue，计数好橘子
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(grid[i][j]==1){
                    count++;
                }
                if(grid[i][j]==2){
                    queue.add(new int[]{i,j});
                }
            }
        }
        int[] dx = new int[]{1,-1,0,0};
        int[] dy = new int[]{0,0,1,-1};
        //广度搜
        while (count>0 && !queue.isEmpty()){//必须有count>0，因为最后队里会有上一次腐化的，又进了一次循环，结尾返回res-1也不行，因为不需要腐化应返回0
            int size = queue.size();
            res++;
            for(int i=0;i<size;i++){
                int[] poll = queue.poll();
                //看周边四个元素，把没腐烂的染烂
                for(int j=0;j<4;j++){
                    int x = poll[0]+dx[j],y = poll[1]+dy[j];
                    if(x>=0 && x<m &&y>=0 && y<n && grid[x][y]==1){
                        queue.add(new int[]{x,y});
                        grid[x][y] = 2;
                        count--;
                    }
                }
            }
        }
        return count==0 ? res : -1;
    }
    public boolean canFinish1(int numCourses, int[][] prerequisites) {
        //拓扑排序问题,广度搜，计算所有节点入度，把入度为0的入队，出队node，将依赖node的节点的入度-1，然后一直进行下去，如果队空时，还有入度不为0的节点，说明无拓扑排序
        Map<Integer, List<Integer>> map = new HashMap<>();//邻接表(key：课号 val：依赖这门课的后续课)
        int[] inDegree = new int[numCourses];//储存每个节点的入度
        for(int i=0;i<numCourses;i++){
            map.put(i,new ArrayList<>());
        }
        //初始化 [a,b]，要修a，必须先修b,所以实际上是 b->a
        for(int[] prereq:prerequisites){
            inDegree[prereq[0]]++;//前面依赖后面，前面入度+1
            map.get(prereq[1]).add(prereq[0]);//很易错，目的是根据b（低级的）去找a（高级的），然后给a的入度-1，所有低级的任务都完成，就能做高级任务，高级入度为0
        }
        //广度搜，借助队列，队里储存入度为0的
        Queue<Integer> queue = new LinkedList<>();
        for(int i=0;i<numCourses;i++){
            if(inDegree[i]==0) queue.add(i);
        }
        int count =0;//计算有几门课能学
        while (!queue.isEmpty()){
            int selected = queue.poll();
            count++;
            List<Integer> list = map.get(selected);//找到依赖这门课的后续课
            for(int i:list){
                inDegree[i]--;//高级的入度-1；
                if(inDegree[i]==0) queue.add(i);
            }
        }
        return count==numCourses;
    }
    class Trie {
        //本质上就是多叉树，每层26个节点，但是不是一次性创建出来，需要一个创建一个
        class TrieNode{
            private boolean isEnd;//用于判断以当前节点为结尾的字符是否存在，比方说abc被插入进去，如果查找ab，应该是不存在，因为b的isEnd是false
            TrieNode[] next;
            public TrieNode() {
                isEnd = false;
                next = new TrieNode[26];
            }
        }
        private TrieNode root;
        public Trie() {
            root = new TrieNode();
        }
        public void insert(String word) {
            //比方说插入字符"abc"，从root开始，去第一层的数组中找节点a，看是否存在，不存在则创建，然后去a的数组中找节点b
            TrieNode node = root;
            for(char c:word.toCharArray()){
                if(node.next[c-'a']==null){
                    node.next[c-'a'] = new TrieNode();
                }
                node = node.next[c-'a'];
            }
            node.isEnd = true;
        }

        public boolean search(String word) {
            TrieNode node = root;
            for(char c:word.toCharArray()){
                node = node.next[c-'a'];
                if(node==null){
                    return false;
                }
            }
            return node.isEnd;//别忘
        }

        public boolean startsWith(String prefix) {
            TrieNode node = root;
            for(char c:prefix.toCharArray()){
                node = node.next[c-'a'];
                if(node==null){
                    return false;
                }
            }
            return true;
        }
    }
    public List<String> letterCombinations(String digits) {
        Map<Character, String> map = new HashMap<>() {{
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
        dfsLetter(res,new StringBuffer(),digits,0,map);
        return res;
    }
    public void dfsLetter(List<String> res,StringBuffer sb,String digits,int index,Map<Character, String> map){
        if(index==digits.length()){
            res.add(sb.toString());
            return;
        }
        char c = digits.charAt(index);
        for (char letter : map.get(c).toCharArray()) {
               sb.append(letter);
               dfsLetter(res,sb,digits,index+1,map);
               sb.deleteCharAt(index);
        }
        return;
    }
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        //同一斜线怎么半？set
        //正常复杂度o(n!),为了优化需要快速判断该位置所在列和两条斜线是否已有皇后
        int[] queens = new int[n];//记录每行皇后所在列数
        Set<Integer> col = new HashSet<Integer>();
        Set<Integer> diag1 = new HashSet<Integer>();
        Set<Integer> diag2 = new HashSet<Integer>();
        dfsQueen(res,queens,0,n,col,diag1,diag2);
        return res;
    }
    public void dfsQueen(List<List<String>> res,int[] queens,int row,int n,
                         Set<Integer> col,Set<Integer> diag1,Set<Integer> diag2){
        if(row==n){
            res.add(generateBoard(queens,n));
            return;
        }
        for(int i=0;i<n;i++){
            //如何描述当前点（row，i）所在的对角线？--row+i(右上到左下)和row-i（左上到右下）
            int d1 = row+i,d2 = row-i;
            if(col.contains(i) || diag1.contains(d1) || diag2.contains(d2)){
                continue;
            }
            col.add(i);diag1.add(d1);diag2.add(d2);
            queens[row] = i;
            dfsQueen(res,queens,row+1,n,col,diag1,diag2);
            col.remove(i);diag1.remove(d1);diag2.remove(d2);
        }
    }
    //根据int[] queens产生结果需要的形式
    public List<String> generateBoard(int[] queens, int n) {
        List<String> res = new ArrayList<>();
        for(int i=0;i<n;i++){
            StringBuffer sb = new StringBuffer();
            for(int j=0;j<n;j++){
                if(j==queens[i]){
                    sb.append("Q");
                    continue;
                }
                sb.append(".");
            }
            res.add(sb.toString());
        }
        return res;
    }
    //旋转排序数组最小值
    public int findMin(int[] nums) {
        //区分旋转排序数组，因为最小值不一定在有序部分
        //比mid和right
        int left =0,right = nums.length-1;
        while (left<=right){//等于？
            int mid = (left+right)/2;
            if(nums[mid]<nums[right]){//等于？
                right = mid;//减1？不能减，mid可能是答案,可不减的话最后剩两个元素，left如果是答案就会一直循环？不会循环，因为两个数的话mid是左
            }else {//if(mid>=right)
                left = mid+1;
            }
        }
        return nums[right];//left?right?
    }
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        //第一反应：两个数组都定位到一半的位置，把较小的那个所在的一半移除，因为一共需要移除m/2+n/2
        //为什么不对？没有站在一个把两个数组当成整体的角度来考虑，应该要找全局第k个，然后分别从两个数组取k/2然后比较
        int m = nums1.length,n = nums2.length;
        int left = (m+n+1)/2;
        int right = (m+n+2)/2;
        return (getKth(nums1,nums2,0,0,left)+getKth(nums1,nums2,0,0,right))*0.5;
    }
    public int getKth(int[] nums1,int[] nums2,int i1,int i2,int k){
        if(i1==nums1.length){
            return nums2[i2+k-1];
        }
        if(i2== nums2.length){
            return nums2[i1+k-1];
        }
        if(k==1){
            return Math.min(nums1[i1],nums2[i2]);
        }
        //if(nums1[i1+k/2-1]>nums2[i2+k/2-1]) 可能越界，没考虑到
        int p1 = Math.min(nums1.length-1 , i1+k/2-1);
        int p2 = Math.min(nums2.length,i2+k/2) - 1;
        if(nums1[p1]<nums2[p2]){
            //把更小的所在的数组的那部分k/2移除，能保证第k个数依然被留下来
            return getKth(nums1, nums2, p1+1, i2, k-(p1-i1+1));
        }else{
            return getKth(nums1, nums2, i1, p2+1, k-(p2-i2+1));
        }
    }
    public boolean isValid(String s) {
        //如果结构有要求怎么办   {[()()],[]},{()}
        //记录上一个闭合的括号，当前闭合中的话上一个不能是大{}？不对当前闭合大上一个不能是中[]，闭合一个大要清除之前的状态 []{}和{[]}
        return false;
    }
    public String decodeString1(String s) {
        //因为涉及到嵌套，先处理内层，再处理外层，所以是栈
        //1.一定要弄清楚栈里存的是什么，有两种解法，1.字母一个栈数字一个站 2.全用一个栈
        // 本质上是一个消去括号的过程，读到 [ 新开一个sb，存之后的字符，读到]，拿到[前面的第一个数字（离当前这组括号最近的数字），得到重复次数，再从sb栈中取出上一组括号内的sb和这个拼接，就得到上一组括号内的全部内容
        //3[a2[cd]] 读a--读2--读[--把2入栈--开一个sb，拼接cd--读到]--栈里取出2--2份cd--栈里取出a，拼接2份cd
        StringBuffer res = new StringBuffer();
        Stack<Integer> numStack = new Stack<>();
        Stack<StringBuffer> sbStack = new Stack<>();
        int num =0;
        for(char c:s.toCharArray()){
            //为什么不先是if(Character.isLetter(c))
            //1.是左括号，把res，前一个数字入栈,重置res记录接下来的字符
            if(c=='['){
                numStack.push(num);
                num=0;//别忘，否则算下一个括号前的num会出错
                sbStack.push(res);
                res = new StringBuffer();
            }
            //2.是右括号，出栈次数，res乘以次数，和之前的res拼接
            else if(c==']'){
                StringBuffer temp = new StringBuffer();
                int preNum = numStack.pop();
                for(int i=0;i<preNum;i++){
                    temp.append(res);
                }
                res = sbStack.pop();
                res.append(temp);
            }
            //3.是数字，记录这个数字，用于之后遇到左括号时入栈，为什么一定要入栈？防止嵌套丢失信息
            else if(c>='0' && c<='9'){//[]前的数字可能是多位数
                num = c-'0'+ num*10;
            }
            //4.其他
            else{
                res.append(c);
            }
        }
        return res.toString();
    }
    //一个栈更好理解了，但是需要翻转字符串或者调用insert头插
    public String decodeString2(String s) {
        Stack<Character> stack = new Stack<>();
        for(char c:s.toCharArray()){
            if(c!=']'){//除了]全入栈
                stack.push(c);
            }
            else {//读到右括号，一直出到与之配对的左括号，然后完成一组拼接，再入栈
                StringBuffer sb = new StringBuffer();
                while (!stack.isEmpty() && stack.peek()!='['){
                    sb.insert(0,stack.pop());//头插，否则需要反转
                }
                String subString = sb.toString();//当前括号内的内容
                //出[
                stack.pop();
                //获取num
                sb = new StringBuffer();
                while (!stack.isEmpty() && Character.isDigit(stack.peek())){
                    sb.insert(0,stack.pop());
                }
                int num = Integer.valueOf(sb.toString());
                for(int i=0;i<num;i++){
                    for(char ch: subString.toCharArray()){
                        stack.push(ch);
                    }
                }
            }
        }
        StringBuffer sb = new StringBuffer();
        while (!stack.isEmpty()){
            sb.insert(0,stack.pop());
        }
        return sb.toString();
    }
    int index=0;//是全局的，因为递归回来之后是接着往下走，而非回来
    public String decodeString3(String s){
        //每层递归的任务都一样，上一层读到左括号会开始下一层递归，新起一个sb，遇到字符直接拼接，遇到数字计算个数，然后下一个一定是左括号，然后开始下一层递归，返回本层括号内的字符内容
        StringBuffer sb = new StringBuffer();
        while (index<s.length()){
            char c = s.charAt(index);
            //if(c=='[') 不可能以左括号开始，即使循环完成一次也是
            if (c==']') {//遇到]说明是上一级的终止，返回
                index++;
                return sb.toString();
            }
            else if (Character.isDigit(c)) {
                int num = 0;
                while (Character.isDigit(s.charAt(index))){
                    num = num*10+ s.charAt(index)-'0';
                    index++;
                }//数字之后必定是[，此时index指向[
                index++;
                String nextS = decodeString3(s);
                for(int i=0;i<num;i++){
                    sb.append(nextS);
                }
            }
            else {//读到字符直接拼接
                index++;
                sb.append(c);
            }
        }
        return sb.toString();
    }
    public String decodeString4(String s) {
        StringBuilder decoded = new StringBuilder();
        int i = 0;
        while (i < s.length()) {
            if (Character.isDigit(s.charAt(i))) {
                int num = 0;
                while (Character.isDigit(s.charAt(i))) {
                    num = num * 10 + (s.charAt(i) - '0');
                    i++;
                }
                // start指向[后面的第一个字符
                int start = i++;
                int count = 1;
                while (count != 0) {
                    if (s.charAt(i) == '[') count++;
                    else if (s.charAt(i) == ']') count--;
                    i++;
                }//i指向本次递归中[对应的]的下一个字符
                String subStr = decodeString4(s.substring(start, i - 1));
                for (int j = 0; j < num; j++) {
                    decoded.append(subStr);
                }
            } else {
                decoded.append(s.charAt(i));
                i++;
            }
        }
        return decoded.toString();
    }
    public int[] dailyTemperatures1(int[] temperatures) {
        //两个栈，一个放温度一个放下标，比栈顶大就出栈，糊涂了，有下标不就有温度了吗
        int[] res = new int[temperatures.length];
        Stack<Integer> stack = new Stack<>();
        for(int i=0;i<temperatures.length;i++){
            while (!stack.isEmpty() && temperatures[i]>temperatures[stack.peek()]){
                int pre = stack.pop();
                res[pre] = i-pre;
            }
            stack.push(i);
        }
        return res;
    }
    public int[] dailyTemperatures(int[] temperatures) {
        //从右向左计算，利用后面已经算出的结果，快速算前面
        int n = temperatures.length;
        int[] res = new int[n];//最后一个结果一定为零
        for(int i=n-2;i>=0;i--){
            for(int j=i+1;j<n;j+=res[j]){//j为第一个可能比i大的
                if(temperatures[i]<temperatures[j]){
                    res[i] = j-i;
                    break;
                }
                else if(res[j]==0) { //t[j]<=t[i],但是已经没有比t[j]更大的了
                    res[i] = 0;
                    break;
                }
            }
        }
        return res;
    }
    public int largestRectangleArea(int[] heights) {
        //在一维数组中对每一个数找到第一个比自己小的元素。这类“在一维数组中找第一个满足某种条件的数”的场景就是典型的单调栈应用场景。
        //遍历所有的位置，以其为高度，找左右第一个比他小的，然后计算出一个可能为结果的面接(right-left-i)*height
        //是一个严格递增栈，为什么一次遍历就能找到结果？一次遍历是怎么快速找到左右最小的
        Stack<Integer> stack = new Stack<>();
        int n= heights.length+2; int res=0;
        //相当于加了哨兵，不用处理栈空和最后仍有元素的情况
        int[] newHeight = new int[n];
        newHeight[0]=0;newHeight[n-1]=0;
        for(int i=1;i<n-1;i++){
            newHeight[i] = heights[i-1];
        }
        heights = newHeight;
        stack.push(0);//先放左哨兵
        for(int i=0;i<n;i++){
            while (heights[stack.peek()]>=heights[i]){//等于的话也要出栈 1.只有严格小才是边界，所以必须保证栈里严格增，2.第二个相等的找到的面积一定更大
                int h = heights[stack.pop()];//需要处理边界，如果栈里只有一个元素，pop出来为高，那左边界应该为-1
                int width = (i-stack.peek()-1);
                res = Math.max(res,width*h);
            }
            stack.push(i);
        }
        return res;
    }





}
