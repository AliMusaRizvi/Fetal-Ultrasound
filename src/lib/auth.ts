import { supabase } from './supabase';
import type { User, Session } from '@supabase/supabase-js';

export interface AppUser {
  id: string;
  email: string;
  full_name: string;
  role: 'doctor' | 'admin';
  doctor_code?: string;
  specialty?: string;
  department?: string;
  phone?: string;
  is_active: boolean;
}

// ─── Sign In ──────────────────────────────────────────────────────────────────
export async function signIn(email: string, password: string): Promise<AppUser> {
  const { data, error } = await supabase.auth.signInWithPassword({ email, password });
  if (error) throw new Error(error.message);
  if (!data.user) throw new Error('Login failed');
  return await fetchProfile(data.user.id);
}

// ─── Sign Out ─────────────────────────────────────────────────────────────────
export async function signOut(): Promise<void> {
  const { error } = await supabase.auth.signOut();
  if (error) throw new Error(error.message);
}

// ─── Get current session ──────────────────────────────────────────────────────
export async function getSession(): Promise<Session | null> {
  const { data } = await supabase.auth.getSession();
  return data.session;
}

// ─── Fetch a user's profile from the profiles table ──────────────────────────
export async function fetchProfile(userId: string): Promise<AppUser> {
  const { data, error } = await supabase
    .from('profiles')
    .select('id, full_name, role, doctor_code, specialty, department, phone, is_active')
    .eq('id', userId)
    .single();

  if (error || !data) throw new Error('Could not load user profile');

  return {
    id: data.id,
    email: '',
    full_name: data.full_name,
    role: data.role as 'doctor' | 'admin',
    doctor_code: data.doctor_code,
    specialty: data.specialty,
    department: data.department,
    phone: data.phone,
    is_active: data.is_active,
  };
}

// ─── Subscribe to auth state changes ─────────────────────────────────────────
export function onAuthChange(callback: (user: AppUser | null) => void) {
  return supabase.auth.onAuthStateChange(async (_event, session) => {
    if (session?.user) {
      try {
        const profile = await fetchProfile(session.user.id);
        profile.email = session.user.email || '';
        callback(profile);
      } catch {
        callback(null);
      }
    } else {
      callback(null);
    }
  });
}

// ─── Admin: Create a new doctor account ──────────────────────────────────────
export async function createDoctorAccount(params: {
  email: string;
  password: string;
  full_name: string;
  specialty?: string;
  department?: string;
  phone?: string;
}): Promise<void> {
  // Use admin invite / create user via service role in a real app
  // For now, use standard signup with doctor role in metadata
  const { data, error } = await supabase.auth.signUp({
    email: params.email,
    password: params.password,
    options: {
      data: {
        full_name: params.full_name,
        role: 'doctor',
      },
    },
  });
  if (error) throw new Error(error.message);
  if (!data.user) throw new Error('Account creation failed');

  // Update profile with additional fields
  await supabase.from('profiles').update({
    specialty: params.specialty,
    department: params.department,
    phone: params.phone,
  }).eq('id', data.user.id);
}
