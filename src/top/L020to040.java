package top;

import java.util.*;
//2月3到2月7
public class L020to040 {
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
     * 21.合并两个有序链表：只想到了递归，没想到迭代
     * 22.括号生成:回溯搜索
     * 23.合并k个升序链表：分治，复杂度时间 knlogk,空间logk。再看
     * 24.两两交换链表中的节点:递归或迭代，迭代不要省内存，把要用的节点都存上
     * 25.k个一组翻转链表:注意细节
     * 26.删除有序数组中的重复项:简单双指针
     * 27.移除元素:思路乱了，实际上也是简单双指针
     * 28.kmp算法：j = next[j]
     * 29.两数相除，不使用乘除取余：快速减算法，注意爆int
     * 30.串联所有单词的字串：滑动窗口
     * 31.下一个排列，理清找到下一个字典序更大的数需要几步，可以从折线图想
     * 32.
     * 33.搜索旋转排序数组：为什么能用二分，怎么用，想折线图，mid左右两侧一定至少有一个是有序的
     * 34.在排序数组中查找元素的第一个和最后一个位置:两次二分，二分得到结果后需要判断是不是target，判断过程需要用到left等，注意是否越界
     * 35.搜索插入位置:二分，找第一个大于等于target的
     * 36.有效的数独：要么时间要么空间，想要一次遍历完成，就要开多个数组储存结果
     * 37.
     * 38.外观数列，没思路,模拟，基本上想了个框架，边界值以及while/for的选择不好,一种双层while加双指针[start,end)，一种for加if [j-count,j)
     * 39.组合总和：使用start（而非user数组），可以先排序，然后剪枝提速，for里面判断减去num[i]后是否小于0，是的话break,一般来说排序问题使用userd数组，组合问题使用start（[2,2,3]和[2,3,2]是否相同）
     * 40.组合总和2：加了个去重，传入下一层的变为i+1
     */
    public ListNode mergeTwoLists1(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode(0);
        ListNode pre = dummy;
        while (list1!=null&&list2!=null){
            if(list1.val<list2.val){
                pre.next = list1;
                list1 = list1.next;
            }else {
                pre.next = list2;
                list2 = list2.next;
            }
            pre = pre.next;
        }
        if(list1==null) pre.next = list2;
        if(list2==null) pre.next = list1;
        return dummy.next;
    }
    public ListNode mergeTwoLists(ListNode list1, ListNode list2){
        if(list1==null) return list2;
        if(list2==null) return list1;
        if(list1.val<list2.val){
            list1.next = mergeTwoLists(list1.next,list2);
            return list1;
        }else {
            list2.next = mergeTwoLists(list2.next,list1);
            return list2;
        }
    }
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        dfsGen(res,new StringBuffer(),0,0,n);
        return res;
    }
    public void dfsGen(List<String> res,StringBuffer sb,int leftCount,int rightCount,int n){
        if(leftCount==n && rightCount==n){
            res.add(sb.toString());
            return;
        }
        if(leftCount>n || rightCount>n || leftCount<rightCount){
            return;
        }
        sb.append('(');
        dfsGen(res,sb,leftCount+1,rightCount,n);
        sb.deleteCharAt(sb.length()-1);
        sb.append(')');
        dfsGen(res,sb,leftCount,rightCount+1,n);
        sb.deleteCharAt(sb.length()-1);
    }
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists==null || lists.length==0) return null;
        return mergeKLists(lists,0,lists.length);
    }
    public ListNode mergeKLists(ListNode[] lists,int left,int right) {
        if(left==right){
            return lists[left];
        }
        //if(left>right) return null;//防止传进来的list为空
        int mid = (right+left)/2;
        ListNode l1 = mergeKLists(lists,left,mid);
        ListNode l2 = mergeKLists(lists,mid+1,right);
        return mergeTwoLists(l1,l2);
    }
    public ListNode swapPairs(ListNode head) {
        if(head==null || head.next==null){
            return head;
        }
        ListNode temp = swapPairs(head.next.next);
        ListNode newhead = head.next;
        head.next.next = head;
        head.next = temp;
        return newhead;
    }
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0,head);
        ListNode pre = dummy;
        while (pre!=null){
            ListNode start = pre.next;//没写出来
            ListNode end = start;
            for(int i=1;i<k && end!=null;i++){//注意条件
                end = end.next;
            }if(end==null) break;//注意条件：start 到 end 的闭区间是k个
            // 存在两个end非空判断，不能只有一个，如果k等于1，就会出现没进for，然后end等于null，然后temp处就会出现空指针
            ListNode temp = end.next;
            end.next = null;
            reverse(start);

            pre.next = end;
            start.next = temp;
            pre = start;
        }
        return dummy.next;
    }
    public ListNode reverse(ListNode head){
        if(head.next==null) return head;
        ListNode newHead = reverse(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
    public int removeDuplicates(int[] nums) {
        int behind = 0;
        for(int i = 1;i<nums.length;i++){
            if(nums[behind]!=nums[i]){
                behind++;
                nums[behind] = nums[i];
                //swap(nums,behind,i); 只在乎不一样那几个元素，数组可变，直接赋值就行
            }
        }
        return behind+1;
    }
    public int removeElement(int[] nums, int val) {
        int j=0;
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=val){
                nums[j] = nums[i];
                j++;
            }
        }
        return j;
    }
    public int strStr(String haystack, String needle) {
        int[] next =  getNext(needle);
        int i=0,j = 0;
        while (i<haystack.length()&&j<needle.length()){
            if(j==-1 || haystack.charAt(i)==needle.charAt(j)){//next[0]为-1;
                i++;
                j++;
            }else {
                j = next[j];
            }
        }
        if(j==needle.length()) return i-j;//返回起始位置
        else return -1;
    }
    public int[] getNext(String s){
        //创建一个与模式字符串长度相同的数组next。
        int[] next = new int[s.length()];
        //next[i]关注的是i之前的字串有多少前缀等于后缀
        //初始化next[0]为-1，i为0，j为-1。
        next[0] = -1;
        int i=0,j=-1;
        while (i<s.length()-1){
            //利用kmp的思想计算next[i+1]，i=1，j=0，当前要计算的就是(0,i]这部分前缀等于后缀的最大的数
            //下面的j看前缀，上面的i看的是后缀
            if(j==-1 ||s.charAt(i)==s.charAt(j)){
                i++;
                j++;
                next[i] = j;
            }else {
                j = next[j];//当前新读到的这个匹配不上，说明next不能直接加1,需要挪动下面的数组达到一个对齐，所以j必定变小
            }
        }
        return next;
    }
    public int divide(int dividend, int divisor) {
        //防越界
        if(divisor == 1) return dividend;
        if(divisor == -1){
            if(dividend>Integer.MIN_VALUE) return -dividend;// 只要不是最小的那个整数，都是直接返回相反数
            return Integer.MAX_VALUE;// 是最小的那个，那就返回最大的整数
        }
        int sign = (dividend>>>31)^(divisor>>>31);////相异为1，说明结果为负数
        if(dividend>=0) dividend = -dividend;//负数转正数存在溢出，所以全转成负数算
        if(divisor>0) divisor = -divisor;
        int res = div(dividend,divisor);
        return sign == 1 ?-res: res;
    }
    public int div(int a,int b){
        if(a>b) return 0;
        //即计算a中有几个b，先计算a中有2^n 个b，然后减掉，递归
        int count =1;
        int tempb = b;
        while (a<=tempb+tempb && tempb+tempb<0){
            tempb = tempb+tempb;
            count = count+count;
        }
        return count+div(a-tempb,b);
    }
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        Map<String,Integer> map = new HashMap<>();
        for(String word:words){
            map.put(word,map.getOrDefault(word,0)+1);
        }
        int oneWordLen = words[0].length();
        int wordCount = words.length;
        // 关键：只需要遍历wordLen中起点:0~wordLen-1，即只需要确定窗口的起始位置就行，因为left在循环中会向后移动
        //使用滑动窗口,统计每个窗口内单词出现的次数, 滑动窗口的起点可以固定位一个单词的长度, 因为一个单词之后的窗口都是重复的
        for(int i=0;i<oneWordLen;i++){//?
            int left=i,right=i,count=0;
            Map<String,Integer> windowMap = new HashMap<>();
            while (right+oneWordLen-1<s.length()){//加完一个单词后的right下标
                String curWord = s.substring(right,right+oneWordLen);
                windowMap.put(curWord,windowMap.getOrDefault(curWord,0)+1);
                right = right + oneWordLen;
                count++;
                // 如果窗口内出现了不满足条件的单词, 我们是需要将其踢出去的, 这个时候就移动左边界
                // 判断条件就是用windowMap和countMap比较, 不满足条件有两种情况,单词不在words数组中,单词出现次数比words数组中的次数多
                while (windowMap.getOrDefault(curWord,0)>map.getOrDefault(curWord,0)){
                    String tempWord = s.substring(left,left+oneWordLen);
                    windowMap.put(tempWord,windowMap.getOrDefault(tempWord,0)-1);
                    count--;
                    left = left+oneWordLen;
                }
                if(count==wordCount) res.add(left);
            }
        }
        return res;
    }
    public void nextPermutation(int[] nums) {
        //想把数变大，一定要把某个高位数字变大
        //1.后面的大数与前面的小数交换2.尽可能靠右，从低位开始找3.剩下的元素升序排列
        int n = nums.length;
        //1.从右往左找，找一个小i，i后面一定存在一个比他大的
        int i = n-2;
        while (i>=0&&nums[i]>=nums[i+1]){
            i--;
        }
        //找到了i，此时nums[i]<nums[i+1],并且i+1之后的元素是单调减的
        if(i<0){
            //可以不用使用sort，因为是单调递减的，只需首尾元素两两交换，然后一直向里
            Arrays.sort(nums);
            return;
        }
        //2.从[i+1,n),找到比i大的中最小那个，直接从右找就行，因为i+1之后是单调减的
        int j = n-1;
        while (nums[j]<=nums[i]){
            j--;
        }
        //3.交换
        swap(nums,i,j);
        //4.i后面剩下的元素升序排雷
        Arrays.sort(nums,i+1,n);
    }
    public int search(int[] nums, int target) {
        int left = 0 ,right = nums.length-1;
        while (left<=right){
            int mid = (left+right)/2;
            //左半有序，看target在不在左半
            if(nums[mid]==target){
                return mid;
            }
            if(nums[left]<=nums[mid]){//这个位置必须有取等 [3,1] target =1;
                // 取等的目的在于保证两部分一定有一部分是有序的，nums[left]==mid意味着两者之间元素全相等
                //只是比了mid和left，不意味着跳到else有mid<right
                if(nums[left]<=target && target<=nums[mid]){
                    right = mid-1;
                }else {
                    left = mid+1;
                }
            }else {
                if(nums[mid]<=target&&target<=nums[right]){
                    left = mid+1;
                }else {
                    right = mid-1;
                }
            }
        }
        return -1;
    }
    public int[] searchRange(int[] nums, int target) {
        if(nums.length==0) return new int[]{-1,-1};
        if(nums.length==1){
            if(nums[0]==target) return new int[]{0,0};
            else return new int[]{-1,-1};
        }
        int left = 0 ,right = nums.length-1;
        int start=-1,end=-1;
        //找第一个等于的,没有则返回-1
        while (left<=right) {
            int mid = (left + right) / 2;
            if (nums[mid] >= target) right = mid - 1;
            else left = mid + 1;
        }
        //return left;
        //因为进if之后需要继续动，然后才发现不符合while条件，退出循环，然后需要访问nums[left],所以可能越界，如[2,2],target=1
        if(left>=0&&left<nums.length&&nums[left]==target) start = left;//判断挺巧妙
        //找最后一个等于的
        left=0;right=nums.length-1;
        while (left<=right){
            int mid = (left+right)/2;
            if(nums[mid]<=target){
                left = mid+1;
            }else right =mid-1;
        }
        if(right>=0&&right<nums.length&&nums[right]==target) end = right;
        return new int[]{start,end};
    }
    public int searchInsert(int[] nums, int target) {
        //找到第一个>=target的
        int left = 0 ,right = nums.length-1;
        while (left<=right){
            int mid = (left+right)/2;
            if(nums[mid]>=target) {
                right = mid-1;
            } else {
                left = mid+1;
            }
        }
        return left;
    }
    public boolean isValidSudoku(char[][] board) {
        //row[i][nums+1]代表第i行中nums出现的次数
        int[][] rows = new int[9][9];
        int[][] columns = new int[9][9];
        int[][][] subBoxes = new int[3][3][9];
        for(int i=0;i<9;i++){
            for(int j=0;j<9;j++){
                if(board[i][j]!='.'){
                    int nums = board[i][j] - '0';
                    rows[i][nums-1]++;
                    columns[j][nums-1]++;
                    subBoxes[i/3][j/3][nums-1]++;
                    //每次写完一个数都要判断
                    if(rows[i][nums-1]>1 || columns[j][nums-1]>1 || subBoxes[i/3][j/3][nums-1]>1){
                        return false;
                    }
                }
            }
        }
        return true;
    }
    public String countAndSay(int n) {
        //是否每次生成新string，怎么把string传到下一层
        String s = "1";
        //1.循环，从n=1开始造数
        for(int i=2;i<=n;i++){
            int start=0,end=0;//双指针确定有多少个一样的数字，完成局部生成
            StringBuffer sb = new StringBuffer();
            while (end<s.length()){
                while (end<s.length() && s.charAt(start)==s.charAt(end)){
                    end++;
                }//[start,end)是相同的
                sb.append(end-start);
                sb.append(s.charAt(start));
                start = end;
            }
            s = sb.toString();
        }
        return s;
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        //Arrays.sort(candidates);用于剪枝
        dfsCombinations(res,new ArrayList<>(),candidates,target,0);
        return res;
    }
    public void dfsCombinations(List<List<Integer>> res,List<Integer> path,int[] candidates,int target,int start){
        if(target==0){
            res.add(new ArrayList<>(path));
        }
        if(target<0){
            return;
        }//注意这个start
        for(int i=start;i<candidates.length;i++){
            //if(target-candidates[i]<0) break;剪枝
            path.add(candidates[i]);
            dfsCombinations(res,path,candidates,target-candidates[i],i);
            path.remove(path.size()-1);
        }
    }
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        dfsCom2(res,new ArrayList<>(),candidates,target,0);
        return res;
    }
    public void dfsCom2(List<List<Integer>> res,List<Integer> path,int[] nums,int target,int start){
        if(target==0){
            res.add(new ArrayList<>(path));
            return;
        }
        if (target<0) return;
        for(int i=start;i<nums.length;i++){
            if(i!=start && nums[i]==nums[i-1]){
                continue;
            }
            if(target-nums[i]<0) break;//剪枝
            path.add(nums[i]);
            dfsCom2(res,path,nums,target-nums[i],i+1);
            path.remove(path.size()-1);
        }
    }







}
