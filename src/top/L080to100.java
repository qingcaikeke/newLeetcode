package top;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Stack;
//213-217
public class L080to100 {
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
     * 81.搜索旋转排序数组 II ,没写明白，各种边界条件，特殊情况，对照 1，加了重复元素后，哪些情况是1处理不了的
     * 82.删除排序链表中的重复元素II,只留下不同的,写的不好
     * 83.删除排序链表中的重复元素,递归（从前往后删或从后往前删），迭代
     * 84.
     * 85.
     * 86.分隔链表,起两个头节点，然后再连到一起就很简单，但注意会别忘了把最后一个large的next指针指向null，否则会出现环，在原链表移动复杂（主要因为头部情况，或者说large头还没出现的情况），
     * 87.
     * 88.合并两个有序数组,简单题，可以复习归并排序
     * 89.格雷编码：归纳法，没思路，如果已经获得了n-1位格雷码，就可以获得n位格雷码
     * 90.子集II:去重没去明白，子集回溯的两种写法，一种选或不选（二叉树），一种数学求所有子集（多叉树）
     * 91.解码方法：动态规划，写的不优雅
     * 92.翻转链表2:简单题，如果拆出来left到right再翻转这之间的部分需要遍历两次，头插法只需要遍历一次
     * 93.复原ip地址:搜索，前导0没处理好
     * 94.二叉树中序遍历:递归，迭代，morris
     * 95.不同的二叉搜索树II:递归没写出来
     * 96.不同的二叉搜索树:动态规划没想到，只想到了递归
     * 97.交错字符串:写的不好，写乱了，边界和优化都没写好，为什么动态规划：重叠子问题，保留计算过的子问题可以提高效率
     * 98.验证二叉搜索树，递归写的不对，对性质的理解有问题，没想到使用中序遍历看递增
     * 99.恢复二叉搜索树:做错了，中序遍历，注意优化if逻辑，或者morris，就是在打res是时候判断当前值和前驱值
     * 100.相同的树:简单递归
     */
    public boolean search(int[] nums, int target) {
        //与33相比，多了重复元素，会影响到程序的时间复杂度吗？最坏时间复杂度变为o(n),所有元素都相等且不等于target
        int left=0,right = nums.length-1;
        while (left<=right){
            int mid = (left+right)/2;
            if(nums[mid]==target) return true;
            if(nums[left]==nums[mid] && nums[mid]==nums[right]){
                //[1,0,1,1,1] 只有一部分相等时认为相等那部分有序，两部分都相等就不知道哪半有序了
                left++;
                right--;
                continue;
            }
            if(nums[left]<=nums[mid]){//原来这个位置有取等是因为两个元素时mid=left，而先前判断过mid≠target，所以需要left=mid+1，如果把取等扔到else里达不到这样的效果
                if(nums[left]<=target && target<=nums[mid]){
                    right = mid-1;
                }else{
                    left = mid+1;
                }
            }else {
                if(nums[mid]<=target && target<=nums[right]){
                    left = mid+1;
                }else{
                    right = mid-1;
                }
            }
        }
        return false;
    }
    public ListNode deleteDuplicates(ListNode head) {
        if(head==null) return head;
        ListNode pre = head;
        ListNode cur = head.next;
        while (cur!=null){
            while (cur!=null && cur.val==pre.val){
                cur = cur.next;
            }
            pre.next = cur;
            pre = cur;
            if(cur!=null)cur = cur.next;
        }
        return head;
    }
    public ListNode deleteDuplicates2(ListNode head) {
        //需要同时关注3个节点 //[1,2,3,3,4,4,5]
        ListNode dummy = new ListNode(0,head);
        ListNode pre = dummy;
        ListNode cur = head;
        while (cur!=null&&cur.next!=null){
            if(cur.val==cur.next.val){
                int value = cur.val;//这种写法比cur.next = cur.next.next;更好
                while (cur!= null && cur.val==value){
                    cur = cur.next;
                }
                pre.next = cur;
            }else {//为什么pre不能轻动？pre指向的一定是一个必定能留下来的元素怒
                pre = cur;
                cur = cur.next;
            }
        }
        return dummy.next;
    }
    //todo 84 85
    public ListNode partition(ListNode head, int x) {
        ListNode SmallDummy = new ListNode(0);
        ListNode LargeDummy = new ListNode(0);
        ListNode SmallPre = SmallDummy;
        ListNode LargePre = LargeDummy;
        ListNode cur = head;
        while (cur!=null){
            if(cur.val<x){
                SmallPre.next = cur;
                SmallPre = SmallPre.next;
            }else{
                LargePre.next = cur;
                LargePre = LargePre.next;
            }
            cur = cur.next;
        }
        LargePre.next = null;//一定别忘，否则会有环
        SmallPre.next = LargeDummy.next;
        return SmallDummy.next;
    }
    //todo 87
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int index = m+n-1;
        int i1 = m-1,i2 = n-1;//指针名不要用i，j这种，pa，pb，p1，p2可以
        while (i1>=0&&i2>=0){
            if(nums1[i1]>nums2[i2]){
                nums1[index--] = nums1[i1--];
            }else {
                nums1[index--] = nums2[i2--];
            }
        }
        while (i2>=0){
            nums1[index--] = nums2[i2--];
        }
    }
    public List<Integer> grayCode(int n) {
        //归纳法,由n-1位格雷码推n位格雷码
        List<Integer> res = new ArrayList<>();
        //将n-1位格雷码，每个高位都加1，这样前半段和后半段都符合格雷码要求
        //然后拼接，为了保证拼接处正确，需要倒序拼接 0a,0b,0c,1c,1b,1a
        res.add(0);
        for(int i=1;i<=n;i++){//i等于1有两个数
            int number = (int) Math.pow(2,i-1);
            for(int j=number-1;j>=0;j--){//找到n-1位格雷码
                res.add(res.get(j)^(1<<i));//高位加1，低位与0异或，原来是啥之后还是啥，注意是i-1
            }
        }
        return res;
    }
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        //对照组合总和与全排类
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        dfsSubsets(false,res,new ArrayList<>(),0,nums);
        return res;
    }
    public void dfsSubsets(boolean choosePre,List<List<Integer>> res,List<Integer> path,int index,int[] nums){
        if(index==nums.length){
            res.add(new ArrayList<>(path));
            return;
        }
        //加一个bool变量，判断前一个元素是否选则
        //若前一个选则，本次随意
        // 如果前一个未选，本次只能不选
        //例如122，选12和选13一样
        //对于当前选择的数x，若前面有与其相同的数y，且没有选择 yyy，此时包含 xxx 的子集，必然会出现在包含 yyy 的所有子集中
        dfsSubsets(false,res,path,index+1,nums);//不加索引为index的元素到子集中
        if(index!=0&&!choosePre&&nums[index]==nums[index-1]){
            return;
        }
        path.add(nums[index]);//加如索引为你index的元素到子集中
        dfsSubsets(true,res,path,index+1,nums);
        path.remove(path.size()-1);
    }
    public void dfsSubsets2(List<List<Integer>> res,List<Integer> path,int index,int[] nums){
        if(index==nums.length){
            return;
        }
        //如123，第一次for选1，第二次for从2开始，相当于不选1
        for(int i = index;i<nums.length;i++){//不需要使用used数组，因为纵向递归进来的i是i+1
            if(i!=index && nums[i]==nums[i-1]){//去重，纵可以，横不行
                continue;
            }
            path.add(nums[i]);
            res.add(new ArrayList<>(path));//不再是二叉树，而是多叉树，可以参考代码随想录的题解，更类似于数学中枚举所有子集的思想
            dfsSubsets2(res,path,i+1,nums);
            path.remove(path.size()-1);
        }
    }
    public int numDecodings(String s) {
        //s的前i位有几种解码方法取决于dp[i-1]+dp[i-2]if(s.[i-1]*s.[i-2]在1-26)
        int n = s.length();
        int[] dp = new int[n+1];
        dp[0] = 1;
        for(int i =1;i<=n;i++){
            int cur = s.charAt(i-1) - '0';
            if(cur!=0) dp[i]+=dp[i-1];
            if(i!=1){
                int pre = s.charAt(i-2) - '0';
                if(pre!=0 && pre*10+cur<=26 ) dp[i]+=dp[i-2];//用当前数和前一个数当成整体去映射，又限制前一个数不为0，则直接小于等于26就行
            }
        }
        return dp[n];
    }
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(0,head);
        ListNode cur = dummy;
        for(int i=1;i<left;i++){
            cur = cur.next;
        }
        ListNode pre = cur; //之后把left到right一次插到pre后面
        cur = cur.next;//现在cur在left
        for(int i=left;i<right;i++){
            ListNode node = cur.next;
            cur.next = cur.next.next;
            node.next = pre.next;
            pre.next = node;
        }
        return dummy.next;
    }
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        dfsIP(res,new ArrayList<>(),0,0,s);
        return res;
    }
    public void dfsIP(List<String> res,List<Integer> path,int start,int count,String s){
        if(start==s.length() && count==4){
            StringBuffer sb  = new StringBuffer();
            for(int i=0;i<4;i++){
                sb.append(path.get(i));
                if(i!=3) sb.append(".");
            }
            res.add(sb.toString());
            return;
        }
        if(start==s.length() || count==4){
            return;
        }
        //如果新的分片第一位是0的话，只有一种选则
        if(s.charAt(start)=='0'){//区分char'0'和0
            path.add(0);
            dfsIP(res,path,start+1,count+1,s);
            path.remove(path.size()-1);
            return;
        }
        int num = 0;
        for(int i=0;i<3&&(start+i)<s.length();i++){
            num = num *10 + s.charAt(start+i)-'0';
            if(num<=255){
                path.add(num);
                dfsIP(res,path,start+i+1,count+1,s);
                path.remove(path.size()-1);
            }
        }
    }
    //94
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        while (root!=null || !stack.isEmpty()){
            while (root!=null){
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            res.add(root.val);
            root = root.right;
        }
        return res;
    }
    public void inOrder(TreeNode root,List<Integer> res){
         if(root==null) return;
         inOrder(root.left,res);
         res.add(root.val);
         inOrder(root.right,res);
    }
    public List<Integer> morris(TreeNode root){
        List<Integer> res = new ArrayList<>();
        TreeNode cur = root;
        //左子树的最右节点
        while (cur!=null){
            if(cur.left!=null) {
                TreeNode node = cur.left;
                while (node.right!=null && node.right!=cur){//条件不要写错
                    node = node.right;
                }//node是左子树的最右节点
                if(node.right!=null){
                    res.add(cur.val);
                    cur = cur.right;
                    node.right = null;
                }else {
                    node.right = cur;
                    cur = cur.left;
                }
            }else {
                res.add(cur.val);
                cur = cur.right;
            }
        }
        return res;
    }
    public List<TreeNode> generateTrees(int n) {
        //递归：有左子树和右子树能生成树，需要生成m*n个故每层放回的应该是一个List<TreeNode>
        return dfsGenerate(1,n);
    }
    public List<TreeNode> dfsGenerate(int start,int end){
        List<TreeNode> res = new ArrayList<>();
        if(start>end) {
            res.add(null);//重点：不是return null；
            return res;
        }
        for(int i=start;i<=end;i++){
            List<TreeNode> leftList = dfsGenerate(start,i-1);
            List<TreeNode> rightList = dfsGenerate(i+1, end);
            for(TreeNode leftNode:leftList){
                for(TreeNode rightNode :rightList){
                    TreeNode root = new TreeNode(i);
                    root.left = leftNode;
                    root.right = rightNode;
                    res.add(root);
                }
            }
        }
        return res;
    }
    public int numTrees(int n) {
        //确定根，左半部分有多少种生成树的方式，右半有多少
        //每层干的一样的事，递归-->发现重复计算，只要n确定，二叉树的个数就是确定的
        int[] dp = new int[n+1]; //dp[1] = 1 dp[2] = 2
        dp[0] = 1;
        for(int i=1;i<=n;i++){//i个节点有几种
            int count=0;
            for(int j=0;j<i;j++){//分配到左边的节点个数,最多i-1个，因为有一个根节点
                count+=dp[j]*dp[i-j-1];
            }
            dp[i] = count;
        }
        return dp[n+1];
    }
    public boolean isInterleave(String s1, String s2, String s3) {
        //dp[i,j] = (dp[i-1][j] &&s3[i+j-1] == s1[i-1]) || (dp[i][j-1] && s3[i+j-1] == s2[j-1])
        //可以理解成一个方阵，要么向下要么向右，因为每次都要比之前多一个元素，要么s2多一个要么s1多一个
        int m = s1.length(),n = s2.length();
        if(m+n!=s3.length()) return false;//别忘
        boolean[] dp = new boolean[m+1];
        dp[0] = true;
        //相当于不用s2
        for(int i=1;i<=m;i++){
            dp[i] = dp[i-1] && s3.charAt(i-1)==s1.charAt(i-1);
        }
        for(int j=1;j<=n;j++){
            dp[0] = s2.charAt(j-1)==s3.charAt(j-1);
            for(int i=1;i<=m;i++){
                dp[i] = (dp[i] && s2.charAt(j-1)== s3.charAt(i+j-1))
                        || (dp[i-1] && s3.charAt(i+j-1)==s1.charAt(i-1));
            }
        }
        return dp[m];
    }
    public boolean isValidBST(TreeNode root) {
        //递归 //想的太简单了，左边右边都是排序树且left.val<root.val<right.val不能说明是排序树，
        // 必须要中大于左边最大的，中间小于右边最小的
        //更好的方法是中序遍历
        return isValidBST(root,Integer.MIN_VALUE,Integer.MAX_VALUE);
    }
    public boolean isValidBST(TreeNode root,int lower,int upper) {
        if(root==null) return true;
        //1.树的所有节点，都需要在范围内，看当前节点是否在low，up范围内，不在false
        if(root.val<=lower || root.val>=upper){
            return false;
        }
        //2.分别看左右子树，并缩小范围
        return isValidBST(root.left, lower, root.val) && isValidBST(root.right, root.val, upper);
    }
    long pre = Long.MIN_VALUE;//中序遍历版
    public boolean inOrder4Valid(TreeNode root){
        if(root==null) return true;
        boolean left = inOrder4Valid(root.left);
        if(!left || root.val<=pre) return false;//左是false就没必要计算右了
        pre = root.val;
        return inOrder4Valid(root.right);
    }
    public void recoverTree(TreeNode root) {
        //中序遍历找到第一个非升序就交换这个方法不行，示例就会出错，
        //因为一个升序列交换两个元素会有一个（相邻交换）或两个（不相邻）位置出现降序
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        TreeNode pre = null;
        TreeNode x = null,y =null;
        while (!stack.isEmpty() || cur!=null){
            while (cur!=null){
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            if(pre!=null && cur.val<=pre.val){
                //只有x，交换x和x的下一个
                //有x有y，x取pre，y取cur
                if(y==null) x = pre;
                y = cur;
            }
            pre = cur;
            cur = cur.right;
        }
        int temp = x.val;
        x.val = y.val;
        y.val = temp;
    }
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p==null && q==null) return true;
        if(p==null || q==null || p.val!=q.val) return false;
        return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);
    }

}
