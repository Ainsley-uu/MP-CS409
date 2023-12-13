import random
def findKthLargest(nums, k):
    def quick_sort(nums, k):
        idx = random.randint(0, len(nums)-1)
        pivot = nums[idx]
        

        left, right = 0, len(nums)-1
        while left < right:
            while left<right and nums[right] < pivot:
                right -= 1
            while left < right and nums[left] >= pivot:
                left += 1
            nums[left], nums[right] = nums[right], nums[left]
        nums[left], nums[idx] = nums[idx], nums[left]
        print("*"*8)
        print(nums, k, left, pivot)
        if left == k:
            return nums[left]
        elif left > k:
            return quick_sort(nums[:left], k)
        else:
            return quick_sort(nums[left+1:], k-left-1)
    return quick_sort(nums, k-1)

print(findKthLargest([3,2,3,1,2,4,5,5,6], 4))