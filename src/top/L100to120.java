package top;

import java.util.*;

public class L100to120 {
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
//    /**
//     * 101.对称二叉树：简单递归,也可以队列，但要注意入队顺序
//     * 102.二叉树层序遍历，简单题
//     * 103.思路本身有误，双端队列没写明白，法1：层序遍历，偶数层反序，法2：双端队列，一次大循环套两次小循环，一次遍历两行
//     * 104.二叉树最大深度：简单递归
//     * 105.前序中序构造二叉树：递归
//     * 106.中序后序构造二叉树：本质上相同
//     * 107.自底向上的层序遍历：还是广度搜索，可以把结果头插法插到list，也可以得到全部结果再reverse
//     * 108.有序数组转换为高度平衡的二叉搜索树:递归
//     * 109.有序链表转二叉搜索树：没写出来
//     * 110.判断一个平衡树是否是二叉树：递归，要注意左子树是平衡树，右是平衡树，合在一起不一定是平衡树，应该计算高度的同时判断是否平衡
//     * 111.二叉树的最小深度：简单递归
//     * 112.路径总和，简单递归，写的不够简洁
//     * 113.路径总和2：搜索
//     * 114.二叉树展开为链表:写了很多次了还没写出来，前驱（正法），栈+根左右，递归（优点反人类）,今天状态不好，明天再看
//     * 115.
//     * 116.填充每个节点的下一个右侧节点指针: 第一反应广度搜，但是因为他是完全二叉树，所以可以递归的方法完成填充
//     * 117.填充每个节点的下一个右侧节点指针2: 广度搜
//     * 118.杨辉三角，模拟
//     * 119.杨辉三角2，近动态规划
//     * 120.三角形最小路径和，进动态规划，注意怎么写的更优雅
//     */
    public boolean isSymmetric(TreeNode root) {
        return check(root.left, root.right);
    }
    public boolean check(TreeNode left,TreeNode right) {
        if(left==null&&right==null) return true;
        if(left==null||right==null||left.val!=right.val) return false;
        return check(left.left,right.right)&&check(left.right,right.left);
    }
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if(root == null) return res;
        queue.add(root);
        while (!queue.isEmpty()){
            int n = queue.size();
            List<Integer> temp = new ArrayList<>();
            for(int i=0;i<n;i++){
                TreeNode node = queue.poll();
                temp.add(node.val);
                if(node.left!=null) queue.add(node.left);
                if(node.right!=null) queue.add(node.right);
            }
            res.add(temp);//直接加temp就行，因为循环里每次都是new一个list
        }
        return res;
    }
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        //可以直接层序遍历，然后加一个bool判断当前行是否需要翻转，  if(leftToRight) Collections.reverse(tempList);
        //也可以temp用双端队列，然后奇数层左到右加进去，偶数层右到左加进去，这样temp里的内容就实现了反序，其他层序遍历不变
        List<List<Integer>> res = new ArrayList<>();
        Deque<TreeNode> deque = new LinkedList<>();
        deque.add(root);
        while(!deque.isEmpty()){
            //奇数右出左入，填加节点先加左
            int n = deque.size();
            List<Integer> temp = new ArrayList<>();
            for(int i=0;i<n;i++){
                TreeNode node = deque.pollFirst();//尾出
                temp.add(node.val);
                if(node.left!=null) deque.addLast(node.left);//头入，先入左子，因为队里当前看的顺序是从左看到右
                if(node.right!=null) deque.addLast(node.right);
            }
            res.add(temp);
            if(n==0) break;//别忘
            //偶数左出右入，填节点先加右，因为是右节点先出
            n = deque.size();
            temp = new ArrayList<>();//一定要new
            for(int i=0;i<n;i++){
                TreeNode node = deque.pollLast();//尾出
                temp.add(node.val);
                if(node.right!=null) deque.addFirst(node.right);//头入，先入右子后入左子
                if(node.left!=null) deque.addFirst(node.left);
            }
            res.add(temp);
        }
        return res;
    }
    public int maxDepth(TreeNode root) {
        if(root.val==0) return 0;
        return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
    }
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = inorder.length;
        Map<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<n;i++){
            map.put(inorder[i],i);
        }
        return buildTree(preorder,inorder,0,n-1,0,n-1,map);
    }
    public TreeNode buildTree(int[] preorder, int[] inorder,int preLeft,int preRight,
                              int inLeft,int inRight,Map<Integer,Integer> map){
        if(preLeft>preRight){
            return null;//注意这块的判别，不能是大于等于然后返回一个新节点，可能会越界
        }
        int rootVal = preorder[preLeft];
        TreeNode root = new TreeNode(rootVal);
        int index = map.get(rootVal);
        int len = index - inLeft;
        TreeNode left = buildTree(preorder,inorder,preLeft+1,preLeft+len,inLeft,index-1,map);
        TreeNode right = buildTree(preorder,inorder,preLeft+len+1,preRight,index+1,inRight,map);
        root.left = left;
        root.right = right;
        return root;
    }
    public TreeNode buildTreePost(int[] inorder, int[] postorder) {
        int n = inorder.length;
        Map<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<n;i++){
            map.put(inorder[i],i);
        }
        return buildTree(inorder,postorder,0,n-1,0,n-1,map);
    }
    public TreeNode buildTreePost(int[] inorder, int[] postorder,
                              int inLeft,int inRight,int postLeft,int postRight,Map<Integer,Integer> map){
        if(inLeft>inRight){
            return null;
        }
        int rootVal = postorder[postRight];//后续遍历中找到根节点
        int index = map.get(rootVal);//根节点点在中序遍历中的位置
        int len =  index - inLeft;//计算左子树长度，注意不用-1
        TreeNode root = new TreeNode(rootVal);
        root.left = buildTree(inorder,postorder,inLeft,index-1,postLeft,postLeft+len-1,map);//注意post的left和right
        root.right = buildTree(inorder,postorder,index+1,inRight,postLeft+len,postRight-1,map);//注意post的left和right
        return root;
    }
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();//一定是link，方便头插
        if(root==null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            int n = queue.size();
            List<Integer> temp = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                TreeNode node = queue.poll();
                temp.add(root.val);
                if(node.left!=null) queue.add(node.left);
                if(node.right!=null) queue.add(node.right);
            }
            res.add(0,temp);
        }
        return res;
    }
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums,0,nums.length-1);
    }
    public TreeNode sortedArrayToBST(int[] nums, int left,int right) {
        if(left>right) return null;
        int mid = (left+right)/2;//任何一棵树，左子树比右子树偶数个少一个节点，奇数个相同
        TreeNode node = new TreeNode(nums[mid]);
        node.left = sortedArrayToBST(nums,left,mid-1);
        node.right = sortedArrayToBST(nums,mid+1,right);
        return node;
    }
    ListNode curr;
    public TreeNode sortedListToBST(ListNode head) {
        //时间o(n)空间，只遍历一次链表，空间o(logn)递归深度，比先存数组再变二叉树更好
        curr = head;
        int len = 0;
        while (head!=null){
            head = head.next;
            len++;
        }
        return helper(0,len-1);
    }
    public TreeNode helper(int left,int right){
        if(left>right) return null;
        //先创建出来，但是不赋值
        TreeNode root = new TreeNode();
        //每次取一半，然后想办法一直往左走，走到没有元素了，就是最左面的节点，给他赋值left的第一个节点
        int mid = (left+right)/2;
        root.left = helper(left,mid-1);
        root.val = curr.val;
        curr = curr.next;
        root.right = helper(mid+1,right);
        return root;
    }
    public boolean isBalanced(TreeNode root) {
        //注意：左是平衡树，右是平衡树，合在一起不一定是平衡树
        //计算高度的同时判断是否平衡
        return getHeight(root) != -1;
    }
    public int getHeight(TreeNode root){
        if(root==null) return 0;
        int leftHeight = getHeight(root.left);
        int rightHeight = getHeight(root.right);
        if(leftHeight==-1 || rightHeight==-1 || Math.abs(leftHeight-rightHeight)>1) return -1;
        return Math.max(leftHeight,rightHeight)+1;
    }

    public int minDepth(TreeNode root) {
        //试试从上往下写呢？
        //是个从下往上的递归
        if(root == null) return 0;
        if(root.left==null||root.right==null) return Math.max(minDepth(root.left),minDepth(root.right))+1;
        return Math.min(minDepth(root.left),minDepth(root.right))+1;
    }
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root.left == null && root.right == null) {
            return targetSum == root.val;
        }
        return hasPathSum(root.left, targetSum - root.val) ||
                hasPathSum(root.right, targetSum - root.val);
    }
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new LinkedList<>();
        dfsPathSum(res,new LinkedList<>(),root,targetSum);
        return res;
    }
    public void dfsPathSum(List<List<Integer>> res, List<Integer> path,TreeNode root ,int target){
        if(root==null) return;
        //结构可以优化，判断完root！=null 之后直接减val
        if(root.left==null && root.right==null ){
            if(target == root.val){
                path.add(root.val);
                res.add(new LinkedList<>(path));
                path.remove(path.size()-1);
                return;
            }
            return;
        }
        path.add(root.val);
        dfsPathSum(res,path,root.left,target-root.val);
        dfsPathSum(res,path, root.right,target-root.val);
        path.remove(path.size()-1);
    }
    //几次都没写好，也可以先前序遍历，存到list然后和展开分开进行
    public void flatten1(TreeNode root) {
        //1.前驱节点法
        TreeNode cur = root;
        while (cur!=null){
            if(cur.left!=null){
                TreeNode node = cur.left;
                while (node.right!=null){
                    node = node.right;
                }//找到左子树的最右节点
                node.right = cur.right;
                cur.right = cur.left;
                cur.left = null;
                cur = cur.right;
            }else {
                cur = cur.right;
            }
        }
    }
    public void flatten2(TreeNode root) {
        if(root==null) return;
        //用栈，根左右的思想
        TreeNode pre = null;//为什么要用pre
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            if(pre!=null){
                pre.right = node;
                pre.left = null;
            }
            pre = node;
            if(node.right!=null) stack.push(node.right);
            if(node.left!=null) stack.push(node.left);//后入先出
        }
    }
    TreeNode pre = null;
    public void flatten(TreeNode root){
        //递归的思想，逆前序遍历，用一个全局pre变量，因为原来的前序遍历会丢失右节点，除了使用迭代的方式把右节点保存到栈里
        //按右左根的顺序走，每次做的事都是把当前节点和pre连起来
        if(root==null) return;
        flatten(root.right);
        flatten(root.left);
        root.right = pre;
        root.left = null;
        pre = root;
    }
    static class ConnectFillRight{
    static class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {}

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    };
    //广度搜，适用于完全二叉树和非完全二叉树
    public Node connect1(Node root) {
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            int n = queue.size();
            for(int i=0;i<n;i++){
                Node poll = queue.poll();
                if(i==n-1) poll.next = null;
                else poll.next  = queue.peek();
                if(poll.left!=null) queue.add(poll.left);
                if(poll.right!=null) queue.add(poll.right);
            }
        }
        return root;
    }
    //递归，通过本层的next指针连接下一层的节点，省略了queue的额外空间
    //也有迭代写法
    public Node connect(Node root) {
        if(root==null || root.left==null) return root;//因为是完全二叉树，没有左就一定没有右
        //每层做什么事？
        //1.把左右子连起来(连左子树的next)
        root.left.next = root.right;
        //2.连右子树的next
        if(root.next!=null) root.right.next = root.next.left;
        //递归的处理左右子节点
        connect(root.left);
        connect(root.right);
        return root;
    }
}
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new LinkedList<>();
        res.add(new LinkedList<>(Arrays.asList(1)));
        for(int i=1;i<numRows;i++){
            List<Integer> temp = new ArrayList<>();
            for(int j=0;j<=i;j++){ //i=1
                //if(j==0||j==i) temp.add(1); 更快，更简单
                int num = 0;
                if(j>=1) num+=res.get(i-1).get(j-1);
                if(j<i)    num+=res.get(i-1).get(j);
                temp.add(num);
            }
            res.add(temp);
        }
        return res;
    }
    public List<Integer> getRow(int rowIndex) {
        List<Integer> pre = new ArrayList<>(List.of(1));
        //List<Integer> cur = pre; 复杂度优化：直接在原链表上操作
        for(int i=1;i<=rowIndex;i++){
            pre.add(0);//延长链表，同时为了计算最后一个
            for(int j = i;j>0;j--){//因为要用j-1和j去更新j，所以只能从后往前
                pre.set(j,pre.get(j-1)+pre.get(j));
            }
        }
        return pre;
    }
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[] dp = new int[n];//不用开辟一个二维数组，直接开一个最长的dp就行，同时这样还比在原链表上操作好理解
        dp[0] = triangle.get(0).get(0);
        for(int i = 1;i<n;i++){
            dp[i] = dp[i-1]+triangle.get(i).get(i);//注意两个特殊位置
            for(int j=i-1;j>0;j--){//一定是从后往前
                dp[j] = Math.min(dp[j],dp[j-1])+triangle.get(i).get(j);
            }
            dp[0] = dp[0]+triangle.get(i).get(0);
        }
        int res = Arrays.stream(dp).min().getAsInt();
        return res;
    }





    



}
