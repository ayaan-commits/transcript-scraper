-- ============================================
-- Supabase Database Setup for Video Transcriber
-- Run this in Supabase SQL Editor
-- ============================================

-- 1. Create profiles table (extends auth.users)
CREATE TABLE IF NOT EXISTS profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT,
    tier TEXT DEFAULT 'free' CHECK (tier IN ('free', 'pro')),
    monthly_limit INTEGER DEFAULT 10,
    monthly_used INTEGER DEFAULT 0,
    reset_date TIMESTAMP WITH TIME ZONE DEFAULT (DATE_TRUNC('month', NOW()) + INTERVAL '1 month'),
    razorpay_customer_id TEXT,
    razorpay_subscription_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Create transcriptions table
CREATE TABLE IF NOT EXISTS transcriptions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    video_url TEXT NOT NULL,
    title TEXT,
    thumbnail TEXT,
    duration FLOAT,
    transcript TEXT,
    summary TEXT,
    language TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Create payments table (for payment history)
CREATE TABLE IF NOT EXISTS payments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    razorpay_payment_id TEXT,
    razorpay_order_id TEXT,
    razorpay_signature TEXT,
    amount INTEGER, -- in paise (29900 = ₹299)
    currency TEXT DEFAULT 'INR',
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'success', 'failed')),
    plan TEXT, -- 'pro_monthly', 'pro_yearly'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Enable Row Level Security (RLS)
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE transcriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE payments ENABLE ROW LEVEL SECURITY;

-- 5. RLS Policies for profiles
CREATE POLICY "Users can view own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = id);

-- 6. RLS Policies for transcriptions
CREATE POLICY "Users can view own transcriptions"
    ON transcriptions FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own transcriptions"
    ON transcriptions FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own transcriptions"
    ON transcriptions FOR DELETE
    USING (auth.uid() = user_id);

-- 7. RLS Policies for payments
CREATE POLICY "Users can view own payments"
    ON payments FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own payments"
    ON payments FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- 8. Function to auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, tier, monthly_limit, monthly_used)
    VALUES (
        NEW.id,
        NEW.email,
        'free',
        10,  -- Free tier: 10 transcriptions/month
        0
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 9. Trigger to create profile on signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 10. Function to reset monthly usage (run via cron)
CREATE OR REPLACE FUNCTION public.reset_monthly_usage()
RETURNS void AS $$
BEGIN
    UPDATE profiles
    SET monthly_used = 0,
        reset_date = DATE_TRUNC('month', NOW()) + INTERVAL '1 month'
    WHERE reset_date <= NOW();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 11. Function to increment usage
CREATE OR REPLACE FUNCTION public.increment_usage(user_uuid UUID)
RETURNS void AS $$
BEGIN
    UPDATE profiles
    SET monthly_used = monthly_used + 1,
        updated_at = NOW()
    WHERE id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 12. Function to upgrade user to Pro
CREATE OR REPLACE FUNCTION public.upgrade_to_pro(user_uuid UUID, razorpay_sub_id TEXT)
RETURNS void AS $$
BEGIN
    UPDATE profiles
    SET tier = 'pro',
        monthly_limit = 100,  -- Pro tier: 100 transcriptions/month
        razorpay_subscription_id = razorpay_sub_id,
        updated_at = NOW()
    WHERE id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 13. Function to downgrade user to Free
CREATE OR REPLACE FUNCTION public.downgrade_to_free(user_uuid UUID)
RETURNS void AS $$
BEGIN
    UPDATE profiles
    SET tier = 'free',
        monthly_limit = 10,
        razorpay_subscription_id = NULL,
        updated_at = NOW()
    WHERE id = user_uuid;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- Done! Your database is now set up.
-- Pricing: Free (10/month), Pro (100/month for ₹299)
-- ============================================
