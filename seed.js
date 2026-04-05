import { createClient } from '@supabase/supabase-js';
import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
dotenv.config({ path: resolve(__dirname, '.env.local') });

const supabaseUrl = process.env.VITE_SUPABASE_URL;
const supabaseKey = process.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error("Missing supabase URL or keys in .env.local");
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function createUser(email, password, role, fullName) {
  console.log(`Creating ${role}: ${email}...`);
  // Use signUp method
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: {
        role: role,
        full_name: fullName
      }
    }
  });

  if (error) {
    if (error.message.includes('already registered')) {
        console.log(`User ${email} already exists.`);
    } else {
        console.error(`Failed to create ${email}:`, error.message);
    }
  } else {
    console.log(`Created user ${email} successfully.`);
    // Attempt to update profiles explicitly just in case trigger takes time or we want extra fields
    if (data?.user?.id) {
       await supabase.from('profiles').update({
           is_active: true
       }).eq('id', data.user.id);
       console.log(`Updated profile for ${email}.`);
    }
  }
}

async function run() {
  await createUser('admin@fetalai.com', 'admin1234', 'admin', 'Admin User');
  await createUser('doctor@fetalai.com', 'doctor1234', 'doctor', 'Dr. Ali');
  console.log("Seeding complete. Note: If email confirmation is required by your Supabase project, you must either click the link in your email or disable 'Confirm Email' in Supabase Auth settings.");
  process.exit(0);
}

run();
